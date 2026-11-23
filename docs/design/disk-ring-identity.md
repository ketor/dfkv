# The in-disk ring and the names of the disks (STO-36)

Status: design, round 5 (review-m3's rounds 1–4 addressed: M1–M3, S1–S4,
N1–N3, R1–R5, Q1–Q3, P1, and the downgrade decision).
No code yet; it follows #541 and its follow-up PR. The engines' on-disk
formats do not change; each cache directory gains one small file.

## 1. The problem

A node spreads keys over its disks with a consistent-hash ring
(`DiskCacheGroup`, `ConHash`, `DFKV_DISK_HASH_WEIGHT` points per disk). Its
members are the `--dir` strings exactly as given. Two spellings of one
directory (`/mnt/disk1/dfkv` and `/mnt/disk1/dfkv/`, a relative path, a path
through a symlinked ancestor) are different members, and so is the same
list with another weight. A restart with a changed spelling, weight or
member set sends keys to other disks; their old copies stay where they were.

Measured on the ring alone (1,000,000 keys, weight 10, `/mnt/diskN/dfkv`):

| Disks | Trailing slash on every dir | On one dir | Weight 10 → 20 | List reversed |
|---|---|---|---|---|
| 2 | 48.9% of keys move | 33.9% | 26.0% | 0% |
| 4 | 74.3% | 30.2% | 38.2% | 0% |
| 6 | 83.1% | 24.1% | 41.5% | 0% |
| 8 | 87.4% | 19.6% | 43.6% | 0% |

A key's home disk (`Filename()` plus a ring lookup) costs about 0.68 µs.

## 2. Why it matters

1. **Misses.** A changed spelling or weight is a partial cold start nobody
   asked for.
2. **A removed value comes back (shown on main).** Run 1 stores K on disk A.
   Run 2, the dirs spelled with a trailing slash, homes K on B: REMOVE finds
   nothing and returns. Run 3, spelled as in run 1, homes K on A again,
   which still holds run 1's value, and serves it. The same holds for a
   weight changed and changed back, and for a member set that drops a disk
   and takes it back (section 3.3).
3. **Takeover leftovers.** With #541, a failed disk's keys go to the other
   disks. A key left on a non-home disk by a ring change can be routed to
   that stale copy by a later failure, the hazard #541 closed for takeovers.

Note on keys that change disk: when one member's name changes, keys move
between every pair of disks (the renamed member's points move, so keys
leave it for its neighbours and arrive from others). Any disk can then hold
misplaced keys, not only the renamed one.

## 3. Proposal

### 3.1 Canonical members

Ring members become canonical ids (`CanonicalDiskId`, introduced by #541:
`weakly_canonical`, trailing `/` removed).
- A start fails if canonicalizing a `--dir` fails (N2), instead of falling
  back to a lexical form that might differ next time.
- The duplicate-directory check compares canonical ids (N3), so two
  spellings of one directory are refused as a duplicate.
- An absolute, normalized path with no symlink in it or in any ancestor is
  its own canonical id (N1). For such a configuration (DEPLOY's example) the
  ring does not change on upgrade. A relative path, a trailing slash or a
  symlinked ancestor changes the member's name once, and its keys move once
  (section 3.6).

### 3.2 The ring record, inside each cache directory

Each cache directory holds `<dir>/.dfkv-ring` (M3): its lifetime is that of
the data, a swapped disk carries its own, and no placement rule is needed.
It holds:
- a format version that covers everything that decides a key's home (S3):
  the ring algorithm (Ketama over MD5), the points per unit of weight, and
  the `Filename()` encoding. Any change to one of them bumps it;
- a node-instance id (R3): 128 random bits, made by the first start that
  writes records on this node and carried by every later one (which id is
  this node's: section 3.3, step 1);
- the weight and the members (canonical ids) in order;
- a run epoch (section 3.3), and a "pass incomplete" flag;
- for a slab disk, the run epoch of its `slab_state` as this start
  activated it (section 3.7).

Every engine and the directory classifier ignore `.dfkv-ring` and its
temporary `.dfkv-ring.tmp` (left by a kill during a write; the next write
replaces it) (R4). Today:
- the log engine refuses unknown names as foreign;
- the file engine refuses anything outside `blocks/` as a different layout;
- slab's fresh-start and resume checks expect an empty directory or their
  own files;
- `ClassifyStoreDir` counts unknown names as "other".

Each check gains the two names. Downgrades: section 3.7.

### 3.3 At every start, before serving

After the disks are opened, in this order:

1. **Read every disk's record, and reset the disks that sat a run out.**
   - **The node's id (Q3, P1)** is the id held by a strict majority of the
     disks that have data ("has data" below); a disk with data and no
     record votes "none". Disks without data do not vote.
     - **No disk has a record:** an upgrade (or a node whose records were all
       removed for a downgrade, section 3.7). Nothing is reset; the pass
       runs (every record counts as missing); the start makes a new id.
     - **An id holds a strict majority:** it is this node's. A data-bearing
       disk with another id came from another node: it is reset, and its
       epoch is not counted.
     - **Some disk has a record, but no id holds a strict majority** (a tie,
       or "none" outnumbers every id): this node's history cannot be told,
       and any disk may have missed runs. Every disk that has data is reset
       (a cold start), not only those with a record: with an older release
       on {A,B,C,D,E}, this release on {A,B} (REMOVE K misses C), and all
       five again, "none" wins 3–2; resetting only A and B would leave C
       serving K. A tie resets both sides: {A,D}, then {A} alone, then
       {A,D} is one id against one "none", and both A and D are reset.
     - Taking "the newest record's id" instead would let a disk moved in
       from a node with a higher epoch take over this node's history.
   - **Disk i is behind (R2, Q1)** if some record of this node newer than
     i's does not list i's canonical id among its members: a start that
     served without i happened since i last served, and REMOVEs of that run
     never reached it. A disk with data and no record counts as epoch 0, so
     it is behind as soon as any record of this node does not list it: an
     older release ran on {A,B,D}, this release on {A,B} (REMOVE K missed
     D), and {A,B,D} again resets D. A behind disk that has data is reset
     to empty (`ResetBeforeActivation`, as #541 resets a failed disk).
     Without this, {A,B,D} → {A,B} (REMOVE K misses) → {A,B,D} would bring
     K back from D, its home in the third run, where a not-home pass keeps
     it.
   - Judging by the member list, not by the epoch alone, means a kill
     between two record writes (some disks at the new epoch, some at the
     old) resets no disk that was a member: no start serves before every
     record is written (step 6).
   - "Has data" is the engine's opened index, before activation (R5): its
     `Count()`. A slab table kept by a warm restart is data; a store reset
     by its own open is not.
   - A disk without a record also counts as missing, which forces the pass
     (step 3).
2. **Activate the disks.**
3. **Decide whether the pass is needed.** It is needed if any record is
   missing (an upgrade, or a disk that lost it), unreadable, of another
   version, carries the "pass incomplete" flag, or describes another ring;
   if a slab disk's `slab_state` epoch moved past its record (section
   3.7); or if #541 found failure records.
4. **The not-home pass** (#541's, on every disk not reset), unless
   `DFKV_DISK_RING_CHANGE=reset`, which resets every disk instead (a cold
   start). Every disk, not only renamed ones: keys move between all disks
   (section 2).
5. **Make the removals durable (S1).** Each engine gets
   `StoreEngine::SyncRemovals()`:
   - log engine: nothing (its REMOVE waits for the journal, `WaitDurable`);
   - slab: nothing (its table survives a process kill in the page cache, and
     a power loss resets the store: the boot id changes);
   - file engine: `syncfs` of the disk, or an fsync of each `blocks/`
     directory it unlinked from (its REMOVE unlinks without a directory
     sync, so after a power loss a removed file can reappear).
6. **Write the record durably (M1):** temporary file, fsync, rename, fsync
   of the cache directory, on every disk of the run.
   - The new epoch is one more than the newest found, with the current
     members, the node id, and each slab disk's activated epoch.
   - If the pass did not complete (a refused removal) or a disk failed
     during the start (S4), the record is written all the same, with the
     "pass incomplete" flag set: the next start runs the pass. Records are
     never deleted (R1): deleting them would erase the epochs that step 1
     needs. In the sequence {A,B,D} (epoch 1) → {A,B} (epoch 2, REMOVE K
     misses D) → {A,B} with an incomplete pass (epoch 3, flagged) →
     {A,B,D}, D's epoch 1 is older than records that do not list D, so D
     is reset and K stays removed.
   - A disk whose record cannot be written is a failed disk (#541's
     degraded mode, or out of placement below the minimum), and goes
     through #541's failure record where that can be written, so that the
     next start empties it.
7. **Serve.**

Writing at step 6 rather than at a clean close closes the kill -9 hole (M1):

- Run 1 on R1 closes with the record R1.
- Run 2 on R2 runs the pass, writes R2, serves, puts K' on B, and is killed.
- Run 3 on R1 finds R2, so it runs the pass. K' (home A under R1) is
  removed from B, and the record becomes R1.
- Run 4 on R2 runs the pass again and finds nothing stale.

The four-run sequence is a test.

### 3.4 Cost of the pass (S2)

- **Walk:** `ForEachKey` over the index and a home lookup per key, about
  0.7 s per million keys, disks in parallel. That is about 0.7 s per TiB at
  1 MiB values and 5.6 s per TiB at 128 KiB slots.
- **Removals:**
  - slab: one record write each;
  - log engine: a REMOVE waits for the journal each time, so removals are
    batched (`StoreEngine::RemoveMany`: the log engine queues every kill and
    waits once; others loop);
  - file engine: one unlink each, plus the sync of step 5.
- **Memory:** keys are collected in chunks (for example 65,536 at a time),
  not one vector per disk.
- **When most keys moved:** if more than half of a disk's keys are
  misplaced, that disk is reset instead of removing them one by one (cold
  for it). The trailing-slash case moves up to 87% of keys (section 1).
- DEPLOY states the one-time cost on upgrade, removals included, and
  `DFKV_DISK_RING_CHANGE=reset` for operators who prefer a fast cold start.

### 3.5 With #541 in the same start

- #541's failure records and the ring record are separate files (decided);
  merging them is a later cleanup.
- A start that finds failure records runs the same pass, and step 6 writes
  the ring record only if the pass completed and no disk failed (S4).
- #541 clears its failure records at a clean close. Its file-engine gap is
  the same as S1, and the #541 follow-up PR fixes it: a file-engine disk's
  removals are synced before the failure record is cleared.

### 3.6 Upgrade

The first start of this release finds no records and runs the pass on
every disk (or resets them, with `DFKV_DISK_RING_CHANGE=reset`). No disk is
behind, since none has an epoch. For the usual configuration (3.1) nothing
is misplaced, and the cost is the walk. Configurations with relative paths,
trailing slashes or symlinked ancestors lose the keys whose home moves,
once.

### 3.7 Downgrades

**(a) The procedure, for every engine.** DEPLOY's downgrade step: remove
`.dfkv-ring` (and any `.dfkv-ring.tmp`) from every cache directory before
starting an older release. The older release then starts as before; the
file engine and the (experimental) log engine would otherwise refuse to
start on the unknown file, which is the safe outcome. A later re-upgrade
finds no records and runs the one-time pass. Member-set changes made while
the older release runs are not protected, as today.

**(b) Defense in depth for slab, the default engine.** `slab_state` holds a
run epoch that every activation bumps (since 2.35.0), whatever the release.
Each slab disk's record stores the epoch this start activated it with (its
value after `MarkDirtyEpoch`).

The next start must compare that with the epoch as the open *found* it,
before any activation (Q2): a store found dirty is activated by its own
constructor (disk_slab_store.cc:510), and `MarkDirtyEpoch` then increments
the epoch, so the value after the open is already one more on every warm
restart. The slab store therefore exposes the epoch `OpenEpochState` read
from `slab_state` (0 for a new store), and the group compares it, with
`!=`, against the recorded one:
- if it differs, some run activated the disk without writing a record (an
  older release, or a run of this one killed between activation and step
  6): the disk's record counts as missing, and the pass runs;
- if it is equal while another slab disk's differs, the disk sat that
  unrecorded run out: it is behind and is reset.

A kill -9 followed by a warm restart finds the recorded epoch: no pass, no
reset. So does a mix of dirty and clean disks.

Remaining false positives, each costing one pass (or one reset), never a
stale value:
- a run of this release killed after activating the disks and before
  writing the records: every slab disk differs, so the pass runs; no disk
  is reset, since all moved;
- a run killed while activating the disks (the group activates them one
  after the other): the disks it reached differ, the others don't, and
  the others are reset although no run served;
- a `slab_state` recreated (a disk replaced under the same path): it
  differs, and the pass runs on a store that is empty anyway.

The node-id rule (section 3.3, step 1) has the same kind of cost: a node
whose disks disagree about their history, which takes a downgrade with
step (a) done on only some disks, or a member set shrunk and grown again
across an upgrade, starts cold. These windows are rare, and each costs
the cache, never a stale value.

So a slab deployment that downgrades without step (a), changes nothing
that reaches the disks, and upgrades again is still safe; a member-set
flip made by the older release is caught by the second rule.

## 4. Settings and metrics

- `DFKV_DISK_RING_CHANGE`: `pass` (default) or `reset`. There is no setting
  that skips the check.
- `dfkv_disks_open_dropped_keys{reason="ring_changed"}`: keys removed by
  the pass for a ring change. Resets are counted under
  `reason="ring_reset"` (behind disks, `reset` mode, the most-keys-moved
  rule).
- An INFO line at start names what changed: members, weight, version,
  behind disks.

## 5. Alternatives

- **Store the ring in each engine's metadata** (`slab_meta`, `lse_super`):
  a format change in two engines for what one file does. Rejected.
- **Refuse to start when the ring changed:** turns a harmless spelling
  change into an outage; dropping misplaced keys is what a cache may do.
  Rejected.
- **Canonical members only, no record:** fixes spellings, but not a weight
  or member set changed and changed back, nor keys left by an earlier
  change. Not enough.

## 6. Tests for the code PR

- The spelling flip-flop (fails on main), with the weight, and the member
  set {A,B,D} → {A,B} → {A,B,D} (the behind-disk reset).
- The four-run kill -9 sequence of 3.3, and R1's four-run sequence with an
  incomplete pass in run 3: 0 resurrections.
- A kill between two record writes: no member disk is reset (R2).
- Node ids (Q3): a disk moved from node P to node Q is reset there, and
  reset again when moved back to P (it then carries Q's id); a disk at
  epoch 50 moved into a node at epoch 5 is reset and does not take the
  node over; a tie (two data disks, one without a record) resets both.
- P1: an older release on {A,B,C,D,E}, this release on {A,B} with REMOVE
  K, then all five: "none" wins 3–2, every data disk is reset, 0
  resurrections; and {A,D} → {A} with REMOVE K → {A,D}: a tie, both reset,
  0 resurrections. No record anywhere (an upgrade): no reset, the pass
  runs.
- Q1's sequence: an older release on {A,B,D}, this release on {A,B} with
  REMOVE K, then {A,B,D}: D (data, no record) is reset, 0 resurrections.
- Q2: kill -9 then a warm restart: no pass, no reset; one dirty and two
  clean slab disks: no reset.
- `.dfkv-ring.tmp` left by a kill: accepted by every engine and by
  `ClassifyStoreDir`, replaced by the next write (R4).
- A slab table kept by a warm restart counts as data for a behind disk
  (R5).
- Upgrade without records; a record that cannot be read, of another
  version, or copies that disagree.
- A refused removal, and a disk that fails during the start: the records
  are written with the "pass incomplete" flag, and the next start runs the
  pass; no record is deleted.
- A record that cannot be written: that disk is treated as failed.
- The file engine's removals are synced before the record is written (a
  seam records the order: the directory syncs come before the rename).
- `DFKV_DISK_RING_CHANGE=reset`; the most-keys-moved reset.
- Every engine and `ClassifyStoreDir` accept `.dfkv-ring`; a fresh slab or
  log directory holding only it initializes.
- Duplicate spellings are refused; a canonicalization error fails the start.
- Slab, log engine and file engine, as #541's tests; the pass's cost at a
  realistic key count (allocator level, no disk).
- Downgrade (b): an older-release run simulated by bumping a slab disk's
  `slab_state` epoch without a record update; the re-upgrade forces the
  pass. And {A,B,D} → downgraded run on {A,B} with REMOVE K (slab_state
  bumped on A and B only, no records) → re-upgrade on {A,B,D}: D is reset,
  0 resurrections.
