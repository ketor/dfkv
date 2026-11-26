#!/usr/bin/env python3
"""Deterministic CLI-35 case-5 workload against real dfkv processes.

Full timeline (seconds, relative to the start of workload traffic):
    0..60 eight nodes, 60 kill -9 n3, 150 start n3 in a new empty
    directory, 240 stop the workload.  Thus down and post-return are each 90 s.

Every variant gets an independent real etcd/MDS/ring.  The run needs only
Linux loopback TCP, the built binaries, an etcd executable (DFKV_ETCD,
ETCD_BIN, or PATH), and an op25-only dfkv_server for ambw-old.  No GPU or
production resource is used.  --smoke shortens only the timeline and is never
an acceptance run.  Runtime data is retained below the supplied output root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable, Iterable
from urllib.request import urlopen

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "integration" / "common" / "src"))
from dfkv_common.cabi import (  # noqa: E402
    DfkvClientOptionsV2,
    make_client_options_v2,
    make_key_array,
)
from dfkv_common.client_metrics import read_native_snapshot  # noqa: E402

ALL_VARIANTS = ("base", "amb", "ambw", "ambw-old", "ambw-now", "ambw-woff")
PREFIXES = 128
BLOCKS = 64
BLOCK_BYTES = 4096
READS_PER_SECOND = 16
NAMESPACE = b"dfkv/cli35/case5/native-v1"
POLL_MS = 1000
MEMORY_MS = 600000
WRITE_ENTRIES = 65536
REMOVE_CANDIDATES = 32
REMOVES_PER_SECOND = 8
REWRITE_DELAY_S = 2
CAP_BYTES = 256 << 20
STATUS_NAMES = {
    0: "OK", 1: "NOT_FOUND", 2: "CACHE_FULL", 3: "QUOTA_EXCEEDED",
    4: "IO_ERROR", 5: "INVALID", 6: "RESOURCE_EXHAUSTED",
    7: "NO_COMPATIBLE_RAIL", 8: "LENGTH_MISMATCH", 9: "PEER_COOLING",
    10: "NO_ROUTE", 11: "BAD_ARGUMENT",
}
CLIENT_ENV_KEYS = (
    "DFKV_RING_ABSENT_GRACE_MS", "DFKV_RING_READBACK_MS",
    "DFKV_RING_READBACK_REWRITE", "DFKV_RING_COOLDOWN_REROUTE",
    "DFKV_RING_REMOVE_FANOUT", "DFKV_RING_REROUTE_KEYS",
    "DFKV_RING_REROUTE_CLOSE_MS", "DFKV_CLIENT_NODE_DEDUP",
    "DFKV_CLIENT_NODE_DEDUP_GPU", "DFKV_RDMA", "DFKV_OPEN_RING_WAIT_MS",
    "DFKV_LOG", "DFKV_GET_MISS_RETRIES", "DFKV_ACCESS_DIR",
)
READBACK_RESULTS = (
    "hit", "miss", "refused_removed", "refused_unknown", "refused_length",
    "refused_gate", "refused_cooling", "hit_rewritten",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(json.dumps(value, sort_keys=True, indent=2,
                                ensure_ascii=False).encode() + b"\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until(origin_ns: int, second: int) -> None:
    remaining = origin_ns + second * 1_000_000_000 - time.monotonic_ns()
    if remaining > 0:
        time.sleep(remaining / 1_000_000_000)


def _until(what: str, predicate: Callable[[], Any], timeout: float = 30.0) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.1)
    raise RuntimeError(f"timed out: {what}; last={last!r}")


class Jsonl:
    """Small locked JSONL sink; operation starts/ends are batch-flushed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("xb")
        self._lock = threading.Lock()
        self._closed = False

    def write(self, value: Any, *, flush: bool = True) -> None:
        self.write_many((value,), flush=flush)

    def write_many(self, values: Iterable[Any], *, flush: bool = True) -> None:
        payload = b"".join(_json_bytes(value) for value in values)
        with self._lock:
            if self._closed:
                raise RuntimeError(f"write to closed evidence log {self.path}")
            self._stream.write(payload)
            if flush:
                self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.flush()
                self._stream.close()
                self._closed = True


@dataclass
class Child:
    name: str
    popen: subprocess.Popen
    log_path: Path
    log_stream: Any
    intentionally_killed: bool = False


class Processes:
    def __init__(self, root: Path, sink: Jsonl) -> None:
        self.root = root
        self.sink = sink
        self.children: list[Child] = []

    def launch(self, name: str, argv: list[str], env: dict[str, str]) -> Child:
        log_path = self.root / "logs" / f"{name}.log"
        log_stream = log_path.open("xb")
        try:
            popen = subprocess.Popen(
                argv, stdout=log_stream, stderr=subprocess.STDOUT,
                cwd=self.root, env=env, close_fds=True,
            )
        except BaseException:
            log_stream.close()
            raise
        child = Child(name, popen, log_path, log_stream)
        self.children.append(child)
        self.sink.write({
            "event": "start", "name": name, "pid": popen.pid, "argv": argv,
            "env": {k: v for k, v in env.items()
                    if k.startswith("DFKV_") or k in ("PATH", "HOME")},
            "wall_time": _iso(), "monotonic_ns": time.monotonic_ns(),
        })
        return child

    def wait_log(self, child: Child, expression: str, timeout: float = 30.0) -> re.Match:
        pattern = re.compile(expression, re.MULTILINE)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = child.log_path.read_text(errors="replace")
            match = pattern.search(text)
            if match:
                return match
            if child.popen.poll() is not None:
                raise RuntimeError(
                    f"{child.name} exited {child.popen.returncode}; log={text[-8000:]}")
            time.sleep(0.05)
        raise RuntimeError(f"{child.name}: no /{expression}/ in {child.log_path}")

    def scenario_kill(self, child: Child) -> None:
        if child.popen.poll() is not None:
            raise RuntimeError(f"{child.name} was already dead: {child.popen.returncode}")
        child.intentionally_killed = True
        child.popen.send_signal(signal.SIGKILL)
        child.popen.wait(timeout=10)
        child.log_stream.flush()
        self.sink.write({"event": "scenario_sigkill", "name": child.name,
                         "pid": child.popen.pid, "returncode": child.popen.returncode,
                         "monotonic_ns": time.monotonic_ns(), "wall_time": _iso()})

    @staticmethod
    def _stop_one(child: Child) -> None:
        if child.popen.poll() is None:
            child.popen.terminate()
            try:
                child.popen.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.popen.kill()
                child.popen.wait(timeout=8)
        child.log_stream.flush()
        child.log_stream.close()

    def close(self) -> None:
        errors = []
        for child in reversed(self.children):
            try:
                self._stop_one(child)
            except Exception as exc:
                errors.append(f"{child.name}: {exc}")
            finally:
                if not child.log_stream.closed:
                    child.log_stream.close()
        if errors:
            raise RuntimeError("; ".join(errors))


@dataclass
class Node:
    node_id: str
    generation: int
    binary_role: str
    binary: Path
    child: Child
    data_dir: Path
    port: int
    metrics_port: int

    @property
    def key(self) -> str:
        return f"{self.node_id}-g{self.generation}"


class Workload:
    def __init__(self) -> None:
        self.keys: list[list[bytes]] = []
        self.values: list[list[bytes]] = []
        self.flat: dict[bytes, int] = {}
        digest = hashlib.sha256()
        for prefix in range(PREFIXES):
            key_row: list[bytes] = []
            value_row: list[bytes] = []
            for block in range(BLOCKS):
                key = f"cli35/case5/p{prefix:03d}/b{block:02d}".encode()
                seed = hashlib.sha256(b"dfkv-cli35-value-v1\0" + key).digest()
                header = b"DFKV-CLI35-CASE5\0" + key + b"\0"
                value = (header + seed * 128)[:BLOCK_BYTES]
                assert len(value) == BLOCK_BYTES
                key_row.append(key)
                value_row.append(value)
                self.flat[key] = prefix * BLOCKS + block
                digest.update(key)
                digest.update(value)
            self.keys.append(key_row)
            self.values.append(value_row)
        self.identity_sha256 = digest.hexdigest()

    def key(self, flat: int) -> bytes:
        return self.keys[flat // BLOCKS][flat % BLOCKS]

    def value(self, flat: int) -> bytes:
        return self.values[flat // BLOCKS][flat % BLOCKS]

    @staticmethod
    def indices(kind: str, second: int, count: int, universe: int) -> list[int]:
        # SHA-256 rather than PRNG implementation details makes every variant's
        # schedule identical and stable across Python versions.
        answer: list[int] = []
        attempt = 0
        while len(answer) < count:
            material = f"dfkv-cli35-case5\0{kind}\0{second}\0{attempt}".encode()
            index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % universe
            if index not in answer:
                answer.append(index)
            attempt += 1
        return answer


class OperationLog:
    def __init__(self, sink: Jsonl, workload: Workload) -> None:
        self.sink = sink
        self.workload = workload
        self._next = 0
        self._lock = threading.Lock()

    def attempt(self) -> int:
        with self._lock:
            self._next += 1
            return self._next

    def starts(self, attempt: int, op: str, keys: list[bytes], *, client: str,
               phase: str, second: int | None, prefix: int | None,
               start_ns: int) -> None:
        rows = []
        for key in keys:
            rows.append({
                "event": "start", "attempt": attempt, "op": op,
                "client": client, "phase": phase, "second": second,
                "prefix": prefix, "flat": self.workload.flat[key],
                "key": key.decode(), "start_ns": start_ns,
                "wall_time_ns": time.time_ns(),
            })
        self.sink.write_many(rows)

    def ends(self, attempt: int, op: str, keys: list[bytes], rows: list[dict], *,
             end_ns: int) -> None:
        output = []
        for key, result in zip(keys, rows):
            output.append({
                "event": "end", "attempt": attempt, "op": op,
                "flat": self.workload.flat[key], "key": key.decode(),
                "end_ns": end_ns, **result,
            })
        self.sink.write_many(output)


class NativeClient:
    def __init__(self, lib: ctypes.CDLL, client_id: str, role: str,
                 mds: str, group: str, operations: OperationLog) -> None:
        self.lib = lib
        self.client_id = client_id
        self.role = role
        self.operations = operations
        options = make_client_options_v2(
            NAMESPACE, mds_endpoints=mds, mds_group=group,
            mds_poll_ms=POLL_MS, register_client=True, client_id=client_id,
            client_info=f"type=cli35-case5;role={role}",
            client_heartbeat_ms=1000, batch_concurrency=8,
        )
        self._options = options
        self.handle = lib.dfkv_open_v2(ctypes.byref(options))
        if not self.handle:
            raise RuntimeError(f"dfkv_open_v2 failed for {client_id}")
        mode = lib.dfkv_transport_mode(self.handle)
        self.transport = mode.decode(errors="replace") if mode else ""
        if not self.transport.startswith("tcp"):
            lib.dfkv_close(self.handle)
            self.handle = None
            raise RuntimeError(f"{client_id}: TCP required, got {self.transport!r}")

    def stats(self) -> str:
        if not self.handle:
            return ""
        return read_native_snapshot(self.lib, self.handle)

    def close(self) -> None:
        if self.handle:
            self.lib.dfkv_close(self.handle)
            self.handle = None

    def _exception_ends(self, attempt: int, op: str, keys: list[bytes],
                        exc: BaseException) -> None:
        self.operations.ends(
            attempt, op, keys,
            [{"error": f"{type(exc).__name__}: {exc}",
              "outcome": "error_or_timeout"} for _ in keys],
            end_ns=time.monotonic_ns())

    def prefix_get(self, prefix: int, *, phase: str, second: int | None) -> dict:
        keys = self.operations.workload.keys[prefix]
        expected = self.operations.workload.values[prefix]
        n = len(keys)
        key_array, key_lengths, key_owners = make_key_array(keys)
        storage = ctypes.create_string_buffer(n * BLOCK_BYTES)
        base = ctypes.addressof(storage)
        pointers = (ctypes.c_void_p * n)(*[base + i * BLOCK_BYTES for i in range(n)])
        capacities = (ctypes.c_uint64 * n)(*([BLOCK_BYTES] * n))
        hits = (ctypes.c_int * n)()
        lengths = (ctypes.c_uint64 * n)()
        statuses = (ctypes.c_uint8 * n)()
        attempt = self.operations.attempt()
        started = time.monotonic_ns()
        self.operations.starts(attempt, "GET", keys, client=self.client_id,
                               phase=phase, second=second, prefix=prefix,
                               start_ns=started)
        call_rc: int | None = None
        try:
            call_rc = self.lib.dfkv_batch_get_auto_st(
                self.handle, key_array, key_lengths, pointers, capacities,
                n, hits, lengths, statuses,
            )
        except BaseException as exc:
            self._exception_ends(attempt, "GET", keys, exc)
            raise
        ended = time.monotonic_ns()
        results: list[dict] = []
        hit_row: list[bool] = []
        mismatch = 0
        for i, key in enumerate(keys):
            status = int(statuses[i])
            hit = call_rc == 0 and hits[i] == 1
            length = int(lengths[i]) if hit else 0
            correct: bool | None = None
            if hit:
                actual = ctypes.string_at(base + i * BLOCK_BYTES, length) if length <= BLOCK_BYTES else b""
                correct = length == BLOCK_BYTES and actual == expected[i]
                if not correct:
                    mismatch += 1
            result = {
                "call_rc": call_rc, "status": status,
                "status_name": STATUS_NAMES.get(status, str(status)),
                "hit": hit, "value_length": length, "payload_correct": correct,
            }
            if call_rc != 0:
                result["error"] = f"dfkv_batch_get_auto_st rc={call_rc}"
            results.append(result)
            hit_row.append(hit)
        self.operations.ends(attempt, "GET", keys, results, end_ns=ended)
        if call_rc != 0:
            raise RuntimeError(f"dfkv_batch_get_auto_st rc={call_rc}")
        first_miss = next((i for i, hit in enumerate(hit_row) if not hit), BLOCKS)
        del key_owners, storage
        return {
            "attempt": attempt, "start_ns": started, "end_ns": ended,
            "first_miss": first_miss, "leading_hits": first_miss,
            "full_hit": first_miss == BLOCKS, "hits": sum(hit_row),
            "payload_mismatches": mismatch,
        }

    def put_keys(self, keys: list[bytes], values: list[bytes], *, phase: str,
                 second: int | None, prefix: int | None) -> list[bool]:
        if not keys:
            return []
        n = len(keys)
        key_array, key_lengths, key_owners = make_key_array(keys)
        storage = ctypes.create_string_buffer(b"".join(values), sum(map(len, values)))
        base = ctypes.addressof(storage)
        pointers: list[int] = []
        offset = 0
        for value in values:
            pointers.append(base + offset)
            offset += len(value)
        pointer_array = (ctypes.c_void_p * n)(*pointers)
        sizes = (ctypes.c_uint64 * n)(*[len(value) for value in values])
        oks = (ctypes.c_int * n)()
        statuses = (ctypes.c_uint8 * n)()
        attempt = self.operations.attempt()
        started = time.monotonic_ns()
        self.operations.starts(attempt, "PUT", keys, client=self.client_id,
                               phase=phase, second=second, prefix=prefix,
                               start_ns=started)
        call_rc: int | None = None
        try:
            call_rc = self.lib.dfkv_batch_put_st(
                self.handle, key_array, key_lengths, pointer_array, sizes,
                n, oks, statuses,
            )
        except BaseException as exc:
            self._exception_ends(attempt, "PUT", keys, exc)
            raise
        ended = time.monotonic_ns()
        results = []
        answer = []
        for i in range(n):
            status = int(statuses[i])
            ok = call_rc == 0 and oks[i] == 1
            row = {
                "call_rc": call_rc, "status": status,
                "status_name": STATUS_NAMES.get(status, str(status)), "ok": ok,
            }
            if call_rc != 0:
                row["error"] = f"dfkv_batch_put_st rc={call_rc}"
            results.append(row)
            answer.append(ok)
        self.operations.ends(attempt, "PUT", keys, results, end_ns=ended)
        del key_owners, storage
        if call_rc != 0:
            raise RuntimeError(f"dfkv_batch_put_st rc={call_rc}")
        return answer

    def prefix_put(self, prefix: int, first: int, *, phase: str,
                   second: int | None) -> list[bool]:
        return self.put_keys(
            self.operations.workload.keys[prefix][first:],
            self.operations.workload.values[prefix][first:],
            phase=phase, second=second, prefix=prefix,
        )

    def flat_put(self, indices: list[int], *, second: int) -> list[bool]:
        workload = self.operations.workload
        return self.put_keys([workload.key(i) for i in indices],
                             [workload.value(i) for i in indices],
                             phase="workload", second=second, prefix=None)

    def _flat_call(self, op: str, indices: list[int], *, second: int,
                   phase: str = "workload") -> list[bool]:
        if not indices:
            return []
        workload = self.operations.workload
        keys = [workload.key(i) for i in indices]
        key_array, key_lengths, key_owners = make_key_array(keys)
        out = (ctypes.c_int * len(keys))()
        attempt = self.operations.attempt()
        started = time.monotonic_ns()
        self.operations.starts(attempt, op, keys, client=self.client_id,
                               phase=phase, second=second, prefix=None,
                               start_ns=started)
        function = self.lib.dfkv_batch_remove if op == "REMOVE" else self.lib.dfkv_batch_exist
        call_rc: int | None = None
        try:
            call_rc = function(self.handle, key_array, key_lengths, len(keys), out)
        except BaseException as exc:
            self._exception_ends(attempt, op, keys, exc)
            raise
        ended = time.monotonic_ns()
        rows = []
        answer = []
        for i in range(len(keys)):
            ok = call_rc == 0 and out[i] == 1
            row: dict[str, Any] = {"call_rc": call_rc, "ok": ok}
            if op == "GET_EXIST":
                row["present"] = ok
            if call_rc != 0:
                row["error"] = f"{function.__name__} rc={call_rc}"
            rows.append(row)
            answer.append(ok)
        self.operations.ends(attempt, op, keys, rows, end_ns=ended)
        del key_owners
        if call_rc != 0:
            raise RuntimeError(f"{op} native batch call rc={call_rc}")
        return answer

    def remove(self, indices: list[int], *, second: int) -> list[bool]:
        return self._flat_call("REMOVE", indices, second=second)

    def exist(self, indices: list[int], *, second: int) -> list[bool]:
        return self._flat_call("GET_EXIST", indices, second=second)


class PerSecond:
    def __init__(self, sink: Jsonl, expected: int) -> None:
        self.sink = sink
        self.expected = expected
        self.rows: dict[int, dict] = {}
        self.lock = threading.Lock()

    def record(self, second: int, result: dict) -> None:
        with self.lock:
            row = self.rows.setdefault(second, {
                "second": second, "prefix_results": 0, "full_prefixes": 0,
                "hit_length_sum": 0, "get_hits": 0,
                "payload_mismatches": 0, "errors": 0, "put_keys": 0,
            })
            row["prefix_results"] += 1
            row["full_prefixes"] += int(result.get("full_hit", False))
            row["hit_length_sum"] += int(result.get("leading_hits", 0))
            row["get_hits"] += int(result.get("hits", 0))
            row["payload_mismatches"] += int(result.get("payload_mismatches", 0))
            row["errors"] += int(bool(result.get("error")))
            row["put_keys"] += int(result.get("put_keys", 0))
            if row["prefix_results"] == self.expected:
                self._flush_locked(row)

    def _flush_locked(self, row: dict) -> None:
        total = row["prefix_results"]
        row["full_prefix_hit_rate"] = row["full_prefixes"] / total
        row["mean_hit_length"] = row["hit_length_sum"] / total
        row["full_prefix_ratio"] = f"{row['full_prefixes']}/{total}"
        row["mean_hit_length_ratio"] = f"{row['hit_length_sum']}/{total}"
        self.sink.write(row)

    def window(self, start: int, end: int) -> dict:
        rows = [self.rows[s] for s in range(start, end) if s in self.rows]
        complete = (len(rows) == end - start and
                    all(row["prefix_results"] == self.expected for row in rows))
        full = sum(row["full_prefixes"] for row in rows)
        total = sum(row["prefix_results"] for row in rows)
        lengths = sum(row["hit_length_sum"] for row in rows)
        result = {
            "start_s": start, "end_s_exclusive": end,
            "expected_seconds": end - start, "observed_seconds": len(rows),
            "complete": complete, "full_prefixes": full, "prefixes": total,
            "hit_length_sum": lengths, "payload_mismatches": sum(
                row["payload_mismatches"] for row in rows),
            "errors": sum(row["errors"] for row in rows),
        }
        if total:
            result.update({
                "full_prefix_hit_rate": full / total,
                "mean_hit_length": lengths / total,
                "full_prefix_ratio_exact": str(Fraction(full, total)),
                "mean_hit_length_ratio_exact": str(Fraction(lengths, total)),
            })
        return result


METRIC_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$")
METRIC_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def _metrics(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    answer = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if not match:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        labels = tuple(sorted((m.group(1), m.group(2).replace('\\"', '"'))
                              for m in METRIC_LABEL.finditer(match.group(2) or "")))
        answer[(match.group(1), labels)] = value
    return answer


def _metric(text: str, name: str, **labels: str) -> float | None:
    wanted = tuple(sorted(labels.items()))
    return _metrics(text).get((name, wanted))


def _metric_sum(text: str, name: str) -> float:
    return sum(value for (metric_name, _), value in _metrics(text).items()
               if metric_name == name)


def _client_core(text: str) -> dict[str, float | None]:
    result = {
        "ring_members": _metric(text, "dfkv_client_ring_members"),
        "registry_gate_open": _metric(text, "dfkv_client_registry_gate_open"),
        "absent_members": _metric(text, "dfkv_client_ring_absent_members"),
        "readback_window_active": _metric(text, "dfkv_client_readback_window_active"),
        "removed_within_ages_fallbacks": _metric(
            text, "dfkv_client_readback_removed_within_ages_fallback_total"),
    }
    for label in READBACK_RESULTS:
        result[f"readback_{label}"] = _metric(
            text, "dfkv_client_readback_reads_total", result=label)
    return result


def _server_core(text: str) -> dict[str, float | None]:
    return {
        "removed_within_ages_requests": _metric(
            text, "dfkv_removed_within_ages_requests_total"),
        "removed_within_queries": _metric_sum(text, "dfkv_removed_within_queries_total"),
        "remove_memory_entries": _metric(text, "dfkv_remove_memory_entries"),
        "write_memory_entries": _metric(text, "dfkv_write_memory_entries"),
        "write_memory_dropped": _metric(text, "dfkv_write_memory_dropped_total"),
        "remove_ok": _metric(text, "dfkv_remove_ok_total"),
        "remove_miss": _metric(text, "dfkv_remove_miss_total"),
    }


class MetricsCollector:
    def __init__(self, run: "VariantRun", origin_ns: int) -> None:
        self.run = run
        self.origin_ns = origin_ns
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="metrics", daemon=True)
        self.capture_lock = threading.Lock()
        self.last_server_raw: dict[str, str] = {}
        self.last_client_raw: dict[str, str] = {}
        self.last_ring: dict = {}
        self.final_complete = False
        self.saw_n3_absent_after_kill = False
        self.saw_n3_return = False

    def start(self) -> None:
        self.thread.start()

    @staticmethod
    def scrape(port: int) -> str:
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1.0) as response:
            return response.read().decode("utf-8", "replace")

    def capture(self, phase: str | None) -> None:
        with self.capture_lock:
            now = time.monotonic_ns()
            second = (now - self.origin_ns) / 1e9
            row: dict[str, Any] = {
                "monotonic_ns": now, "workload_second": second,
                "phase": phase, "clients": {}, "servers": {},
            }
            for client in self.run.client_snapshot():
                try:
                    text = client.stats()
                    self.last_client_raw[client.client_id] = text
                    row["clients"][client.client_id] = _client_core(text)
                    if phase:
                        path = self.run.root / "metrics" / f"client-{client.client_id}-{phase}.prom"
                        path.write_text(text)
                except Exception as exc:
                    row["clients"][client.client_id] = {"error": str(exc)}
            for node in self.run.node_snapshot():
                try:
                    text = self.scrape(node.metrics_port)
                    self.last_server_raw[node.key] = text
                    row["servers"][node.key] = _server_core(text)
                    if phase:
                        path = self.run.root / "metrics" / f"server-{node.key}-{phase}.prom"
                        path.write_text(text)
                except Exception as exc:
                    row["servers"][node.key] = {"error": str(exc)}
            if phase == "final":
                self.final_complete = all(
                    "error" not in values
                    for group in (row["clients"], row["servers"])
                    for values in group.values())
            self.run.sinks["metrics"].write(row)
            for key, text in self.last_server_raw.items():
                path = self.run.root / "metrics" / f"server-{key}-last.prom"
                path.write_text(text)
            if not phase:
                return
            for client_id, text in self.last_client_raw.items():
                path = self.run.root / "metrics" / f"client-{client_id}-last.prom"
                path.write_text(text)

    def poll_ring(self) -> None:
        record: dict[str, Any] = {"monotonic_ns": time.monotonic_ns(),
                                  "wall_time": _iso()}
        try:
            ring = self.run.ring()
            record["members"] = [
                {"id": m["id"], "ip": m["ip"], "port": m["port"],
                 "eligible": m["eligible"]} for m in ring["members"]]
            self.last_ring = ring
            ids = {m["id"] for m in ring["members"]}
            if self.run.kill_done and "n3" not in ids:
                self.saw_n3_absent_after_kill = True
            if self.run.restart_done and "n3" in ids:
                self.saw_n3_return = True
        except Exception as exc:
            record["error"] = str(exc)
        self.run.sinks["membership"].write(record)

    def _loop(self) -> None:
        next_second = 0
        while not self.stop_event.is_set():
            _wait_until(self.origin_ns, next_second)
            if self.stop_event.is_set():
                break
            self.poll_ring()
            self.capture(None)
            next_second += 1

    def stop_and_join(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("metrics collector did not stop")


class Remover:
    def __init__(self, run: "VariantRun", client: NativeClient, number: int,
                 start_second: int) -> None:
        self.run = run
        self.client = client
        self.number = number
        self.start_second = start_second
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop,
                                       name=f"remover-{number}", daemon=True)
        self.pending_rewrite: dict[int, list[int]] = {}
        self.removed = 0
        self.rewritten = 0

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        try:
            for second in range(self.start_second, self.run.timing["end_s"]):
                _wait_until(self.run.origin_ns, second)
                if self.stop_event.is_set():
                    break
                rewrite = self.pending_rewrite.pop(second, [])
                if rewrite:
                    self.client.flat_put(rewrite, second=second)
                    self.rewritten += len(rewrite)
                candidates = Workload.indices(
                    f"remover-{self.number}", second, REMOVE_CANDIDATES,
                    PREFIXES * BLOCKS)
                present_flags = self.client.exist(candidates, second=second)
                present = [key for key, exists in zip(candidates, present_flags) if exists]
                present = present[:REMOVES_PER_SECOND]
                if not present:
                    continue
                self.client.remove(present, second=second)
                self.removed += len(present)
                due = second + REWRITE_DELAY_S
                if due < self.run.timing["end_s"]:
                    self.pending_rewrite.setdefault(due, []).extend(
                        key for key in present if (key + self.number) % 2 == 0)
                self.run.sinks["churn"].write({
                    "second": second, "remover": self.number,
                    "client": self.client.client_id, "candidates": candidates,
                    "present_removed": present, "rewrite_due_s": due,
                    "rewrite_keys": [key for key in present if (key + self.number) % 2 == 0],
                })
        except BaseException as exc:
            self.run.errors.put(f"remover-{self.number}: {type(exc).__name__}: {exc}")

    def stop_and_join(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join()
        if self.thread.is_alive():
            raise RuntimeError(f"remover-{self.number} did not stop")


class VariantRun:
    def __init__(self, args: argparse.Namespace, variant: str,
                 timing: dict[str, int], workload: Workload, etcd: Path) -> None:
        self.args = args
        self.variant = variant
        self.timing = timing
        self.workload = workload
        self.etcd = etcd
        self.root = args.output_root / variant
        self.root.mkdir()
        (self.root / "logs").mkdir()
        (self.root / "metrics").mkdir()
        (self.root / "client-access").mkdir()
        (self.root / "server-access").mkdir()
        self.sinks = {
            name: Jsonl(self.root / f"{name}.jsonl")
            for name in ("operations", "per_second", "prefix_results", "timeline",
                         "metrics", "membership", "processes", "churn")
        }
        self.processes = Processes(self.root, self.sinks["processes"])
        self.operations = OperationLog(self.sinks["operations"], workload)
        self.errors: queue.Queue[str] = queue.Queue()
        self.clients: dict[str, NativeClient] = {}
        self.clients_lock = threading.Lock()
        self.nodes: list[Node] = []
        self.retired_nodes: list[Node] = []
        self.nodes_lock = threading.Lock()
        self.loader: NativeClient | None = None
        self.reader: NativeClient | None = None
        self.removers: list[Remover] = []
        self.executor: concurrent.futures.ThreadPoolExecutor | None = None
        self.metrics: MetricsCollector | None = None
        self.stats = PerSecond(self.sinks["per_second"], READS_PER_SECOND)
        self.origin_ns = 0
        self.kill_done = False
        self.restart_done = False
        self.kill_evidence: dict = {}
        self.restart_evidence: dict = {}
        self.result: dict[str, Any] = {
            "variant": variant, "acceptance_eligible": not args.smoke,
            "smoke": args.smoke, "timing": timing,
            "workload": {
                "prefixes": PREFIXES, "blocks_per_prefix": BLOCKS,
                "block_bytes": BLOCK_BYTES, "initial_mib": 32,
                "sampled_prefixes_per_second": READS_PER_SECOND,
                "workload_identity_sha256": workload.identity_sha256,
                "reader_schedule": "SHA-256 kind=reader, 16 distinct prefixes/second",
                "churn": "existent-key probe; up to 8 REMOVEs/s/remover; half PUT again after 2 s",
            },
            "client_env": self.client_env(),
            "server_env": self.server_env(current=True),
            "status": "RUNNING", "errors": [],
        }
        self.group = f"cli35-{variant}-{os.getpid()}"
        self.mds_endpoint = ""
        self.etcd_endpoint = ""
        self._saved_env: dict[str, str | None] = {}
        self._final_captured = False
        self._correctness: dict = {}
        self._cleaned_up = False

    def client_env(self) -> dict[str, str]:
        enabled = self.variant != "base"
        rewrite = self.variant in ("ambw", "ambw-old", "ambw-now")
        return {
            "DFKV_RING_ABSENT_GRACE_MS": str(MEMORY_MS if enabled else 0),
            "DFKV_RING_READBACK_MS": str(MEMORY_MS if enabled else 0),
            "DFKV_RING_READBACK_REWRITE": "1" if rewrite else "0",
            "DFKV_RING_COOLDOWN_REROUTE": "0",
            "DFKV_RING_REMOVE_FANOUT": "3",
            "DFKV_RING_REROUTE_KEYS": "65536",
            "DFKV_RING_REROUTE_CLOSE_MS": "2000",
            "DFKV_CLIENT_NODE_DEDUP": "0",
            "DFKV_CLIENT_NODE_DEDUP_GPU": "0",
            "DFKV_RDMA": "0",
            "DFKV_OPEN_RING_WAIT_MS": "30000",
            "DFKV_GET_MISS_RETRIES": "1",
            "DFKV_LOG": "WARN",
            "DFKV_ACCESS_DIR": str(self.root / "client-access"),
        }

    def server_env(self, *, current: bool, old: bool = False) -> dict[str, str]:
        enabled = self.variant != "base"
        # 'now' is all-server-W-off; 'old' has W only on new binaries.
        server_w = self.variant in ("ambw", "ambw-old", "ambw-woff")
        write_entries = WRITE_ENTRIES if enabled and server_w and current and not old else 0
        return {
            "DFKV_REMOVE_MEMORY_MS": str(MEMORY_MS if enabled else 0),
            "DFKV_REMOVE_MEMORY_ENTRIES": "1048576",
            "DFKV_WRITE_MEMORY_ENTRIES": str(write_entries),
            "DFKV_LOG": "INFO",
            "DFKV_RDMA": "0",
            "DFKV_ACCESS_DIR": str(self.root / "server-access"),
        }

    def apply_client_env(self) -> None:
        wanted = self.client_env()
        for key in CLIENT_ENV_KEYS:
            self._saved_env[key] = os.environ.get(key)
            os.environ[key] = wanted[key]
        self.result["client_env"] = wanted

    def restore_client_env(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._saved_env.clear()

    def client_snapshot(self) -> list[NativeClient]:
        with self.clients_lock:
            return list(self.clients.values())

    def node_snapshot(self) -> list[Node]:
        with self.nodes_lock:
            return list(self.nodes)

    def all_node_generations(self) -> list[Node]:
        with self.nodes_lock:
            return [*self.retired_nodes, *self.nodes]

    def open_client(self, role: str, number: int) -> NativeClient:
        client_id = f"{self.variant}-{role}-{number}-{os.getpid()}"
        client = NativeClient(self.args.lib_handle, client_id, role, self.mds_endpoint,
                              self.group, self.operations)
        with self.clients_lock:
            self.clients[client_id] = client
        self.sinks["timeline"].write({
            "event": "client_open", "client": client_id, "role": role,
            "transport": client.transport, "monotonic_ns": time.monotonic_ns(),
        })
        return client

    def close_client(self, client: NativeClient) -> None:
        with self.clients_lock:
            self.clients.pop(client.client_id, None)
        client.close()

    def start_etcd(self) -> None:
        for attempt in range(1, 6):
            client_port, peer_port = _free_port(), _free_port()
            data = self.root / "runtime" / f"etcd-attempt-{attempt}"
            data.mkdir(parents=True)
            client_url = f"http://127.0.0.1:{client_port}"
            peer_url = f"http://127.0.0.1:{peer_port}"
            argv = [
                str(self.etcd), "--data-dir", str(data),
                "--name", f"cli35-{self.variant}",
                "--listen-client-urls", client_url,
                "--advertise-client-urls", client_url,
                "--listen-peer-urls", peer_url,
                "--initial-advertise-peer-urls", peer_url,
                "--initial-cluster", f"cli35-{self.variant}={peer_url}",
            ]
            child = self.processes.launch(f"etcd-{attempt}", argv, dict(os.environ))
            self.etcd_endpoint = f"127.0.0.1:{client_port}"
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if child.popen.poll() is not None:
                    break
                try:
                    with urlopen(client_url + "/health", timeout=0.5) as response:
                        health = json.loads(response.read())
                    if health.get("health") == "true":
                        self.result["etcd"] = {
                            "executable": str(self.etcd), "endpoint": self.etcd_endpoint,
                            "data_dir": str(data), "real_lease_expiry": True,
                        }
                        return
                except Exception:
                    pass
                time.sleep(0.1)
            if child.popen.poll() is None:
                raise RuntimeError(f"etcd did not become healthy; see {child.log_path}")
        raise RuntimeError("etcd failed on five independent port/data-dir attempts")

    def start_mds(self) -> None:
        child = self.processes.launch(
            "mds", [str(self.args.mds), "--listen", "0",
                    "--etcd", self.etcd_endpoint, "--etcd-probe-ms", "10000",
                    "--metrics-port", "0", "--metrics-bind", "127.0.0.1"],
            {**os.environ, "DFKV_LOG": "INFO"},
        )
        match = self.processes.wait_log(child, r"^dfkv_mds listening on (\d+),")
        self.mds_endpoint = f"127.0.0.1:{int(match.group(1))}"
        self.result["mds"] = {"executable": str(self.args.mds),
                              "endpoint": self.mds_endpoint, "group": self.group}

    def start_node(self, number: int, generation: int, phase: str) -> Node:
        node_id = f"n{number}"
        binary_role = "old" if self.variant == "ambw-old" and number == 4 else "current"
        binary = self.args.old_server if binary_role == "old" else self.args.server
        for attempt in range(1, 6):
            port = _free_port()
            data = self.root / "runtime" / f"{node_id}-{phase}-attempt-{attempt}"
            data.mkdir(parents=True)
            env = {**os.environ, **self.server_env(
                current=binary_role == "current", old=binary_role == "old")}
            argv = [
                str(binary), "--dir", str(data), "--port", str(port),
                "--cap", str(CAP_BYTES), "--store-engine", "file",
                "--mds", self.mds_endpoint, "--group", self.group,
                "--id", node_id, "--advertise", f"127.0.0.1:{port}",
                "--metrics-port", "0", "--metrics-bind", "127.0.0.1",
                "--mds-registration-timeout-ms", "10000",
            ]
            child = self.processes.launch(f"{node_id}-g{generation}-a{attempt}", argv, env)
            try:
                port_match = self.processes.wait_log(child, r"^PORT (\d+)\s*$")
                if int(port_match.group(1)) != port:
                    raise RuntimeError(f"{node_id}: listened on an unexpected port")
                metrics_match = self.processes.wait_log(
                    child, r"dfkv_server /metrics on port (\d+)")
                return Node(node_id, generation, binary_role, binary, child, data,
                            port, int(metrics_match.group(1)))
            except RuntimeError:
                if child.popen.poll() is None:
                    raise
                text = child.log_path.read_text(errors="replace")
                if f"failed to start on port {port}" not in text:
                    raise
        raise RuntimeError(f"{node_id}: no usable port in five attempts")

    def ring(self) -> dict:
        completed = subprocess.run(
            [str(self.args.ctl), "ring", "--json", "--allow-empty",
             "--group", self.group, "--mds", self.mds_endpoint],
            text=True, capture_output=True, timeout=8, check=True,
        )
        return json.loads(completed.stdout)

    def wait_initial_membership(self) -> None:
        expected = {f"n{i}" for i in range(8)}

        def ready() -> bool:
            try:
                members = self.ring()["members"]
                return {m["id"] for m in members} == expected and all(
                    m["eligible"] for m in members)
            except Exception:
                return False

        _until("eight real MDS members", ready, 40)

    def kill_n3(self) -> None:
        started = time.monotonic_ns()
        with self.nodes_lock:
            old = self.nodes[3]
            self.retired_nodes.append(old)
        self.processes.scenario_kill(old.child)
        self.kill_done = True
        self.kill_evidence = {
            "planned_second": self.timing["kill_s"], "actual_start_ns": started,
            "workload_second": (started - self.origin_ns) / 1e9,
            "node": "n3", "returncode": old.child.popen.returncode,
            "retained_data_dir": str(old.data_dir), "generation": old.generation,
        }
        self.sinks["timeline"].write({"event": "n3_killed_empty_return_planned",
                                      **self.kill_evidence})

    def restart_n3(self) -> None:
        started = time.monotonic_ns()
        replacement = self.start_node(3, 1, "return-empty")
        with self.nodes_lock:
            self.nodes[3] = replacement
        self.restart_done = True
        self.restart_evidence = {
            "planned_second": self.timing["restart_s"], "actual_start_ns": started,
            "workload_second": (started - self.origin_ns) / 1e9,
            "node": "n3", "generation": replacement.generation,
            "fresh_empty_data_dir": str(replacement.data_dir),
            "killed_data_dir_retained": self.kill_evidence.get("retained_data_dir"),
            "fresh_directory_distinct": str(replacement.data_dir) != self.kill_evidence.get(
                "retained_data_dir"),
        }
        self.sinks["timeline"].write({"event": "n3_returned_empty",
                                      **self.restart_evidence})

    def wait_client_gate(self) -> None:
        expected = {client.client_id for client in self.client_snapshot()}

        def ready() -> bool:
            clients = self.client_snapshot()
            try:
                snapshots = {client.client_id: client.stats() for client in clients}
            except Exception:
                return False
            if any(_metric(text, "dfkv_client_ring_members") != 8
                   for text in snapshots.values()):
                return False
            if self.variant != "base" and any(
                    _metric(text, "dfkv_client_registry_gate_open") != 1
                    for text in snapshots.values()):
                return False
            try:
                output = subprocess.run(
                    [str(self.args.ctl), "clients", "--group", self.group,
                     "--mds", self.mds_endpoint], text=True, capture_output=True,
                    timeout=8, check=True).stdout
            except Exception:
                return False
            return all(client_id in output for client_id in expected)

        _until("client MDS registrations and registry gate", ready, 40)
        self.sinks["timeline"].write({
            "event": "initial_clients_registered_gate_ready",
            "clients": sorted(expected), "monotonic_ns": time.monotonic_ns(),
        })

    def read_task(self, second: int, prefix: int) -> None:
        result: dict[str, Any] = {}
        try:
            assert self.reader is not None
            get = self.reader.prefix_get(prefix, phase="workload", second=second)
            result.update(get)
            if get["first_miss"] < BLOCKS:
                oks = self.reader.prefix_put(prefix, get["first_miss"],
                                             phase="workload", second=second)
                result["put_keys"] = len(oks)
                result["put_ok"] = sum(oks)
        except BaseException as exc:
            result.update({
                "error": f"{type(exc).__name__}: {exc}", "full_hit": False,
                "leading_hits": 0, "hits": 0, "payload_mismatches": 0,
            })
            self.errors.put(f"reader second={second} prefix={prefix}: {exc}")
        result.update({"second": second, "prefix": prefix,
                       "client": self.reader.client_id if self.reader else ""})
        self.sinks["prefix_results"].write(result)
        self.stats.record(second, result)

    def check_errors(self) -> None:
        try:
            error = self.errors.get_nowait()
        except queue.Empty:
            return
        others = []
        while True:
            try:
                others.append(self.errors.get_nowait())
            except queue.Empty:
                break
        raise RuntimeError("; ".join([error, *others]))

    def join_readers(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def execute(self) -> None:
        self.start_etcd()
        self.start_mds()
        for number in range(8):
            node = self.start_node(number, 0, "killed" if number == 3 else "initial")
            self.nodes.append(node)
        self.wait_initial_membership()
        self.sinks["timeline"].write({
            "event": "initial_ring_ready", "members": self.ring()["members"],
            "monotonic_ns": time.monotonic_ns(),
        })
        self.apply_client_env()
        loader = self.open_client("loader", 0)
        self.loader = loader
        for prefix in range(PREFIXES):
            oks = loader.prefix_put(prefix, 0, phase="initial", second=None)
            if not all(oks):
                raise RuntimeError(f"initial load failed at prefix {prefix}: {oks}")
        self.result["initial_load"] = {
            "keys": PREFIXES * BLOCKS, "bytes": PREFIXES * BLOCKS * BLOCK_BYTES,
            "all_puts_ok": True, "scope_excluded_from_violation_put_witnesses": True,
        }
        self.close_client(loader)
        self.loader = None
        self.reader = self.open_client("reader", 0)
        remover0_client = self.open_client("remover", 0)
        self.wait_client_gate()

        self.origin_ns = time.monotonic_ns()
        self.result["workload_origin"] = {
            "monotonic_ns": self.origin_ns, "wall_time_ns": time.time_ns(),
            "wall_time": _iso(),
        }
        self.metrics = MetricsCollector(self, self.origin_ns)
        self.metrics.capture("baseline")
        self.metrics.start()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=READS_PER_SECOND, thread_name_prefix="case5-reader")
        remover0 = Remover(self, remover0_client, 0, 0)
        self.removers.append(remover0)
        remover0.start()
        self.sinks["timeline"].write({
            "event": "workload_start", "monotonic_ns": self.origin_ns,
            "planned": self.timing,
        })
        for second in range(self.timing["end_s"]):
            _wait_until(self.origin_ns, second)
            if second == self.timing["kill_s"]:
                self.kill_n3()
            if second == self.timing["restart_s"]:
                self.restart_n3()
            if second == self.timing["second_remover_s"]:
                client = self.open_client("remover", 1)
                remover1 = Remover(self, client, 1, second)
                self.removers.append(remover1)
                remover1.start()
                self.sinks["timeline"].write({
                    "event": "fresh_second_remover_started", "second": second,
                    "client": client.client_id, "monotonic_ns": time.monotonic_ns(),
                })
            assert self.executor is not None
            for prefix in Workload.indices("reader", second, READS_PER_SECOND, PREFIXES):
                self.executor.submit(self.read_task, second, prefix)
            self.check_errors()
        _wait_until(self.origin_ns, self.timing["end_s"])
        self.join_readers()
        for remover in self.removers:
            remover.stop_and_join()
        self.check_errors()
        self.metrics.stop_and_join()
        self.metrics.capture("final")
        self._final_captured = True
        self.metrics.poll_ring()
        self._collect_results()

    def _collect_results(self) -> None:
        restart = self.timing["restart_s"]
        if self.args.smoke:
            unavailable = {"complete": False, "available": False,
                           "reason": "window exceeds shortened smoke timeline; non-acceptance"}
            self.result["core"] = {
                "first_10s_post_return": dict(unavailable),
                "first_30s_post_return": dict(unavailable),
                "post_return_30_60s": dict(unavailable),
                "whole_run": self.stats.window(0, self.timing["end_s"]),
            }
        else:
            self.result["core"] = {
                "first_10s_post_return": self.stats.window(restart, restart + 10),
                "first_30s_post_return": self.stats.window(restart, restart + 30),
                "post_return_30_60s": self.stats.window(restart + 30, restart + 60),
                "whole_run": self.stats.window(0, self.timing["end_s"]),
            }
        self.result["kill_evidence"] = self.kill_evidence
        self.result["restart_evidence"] = self.restart_evidence
        self.result["membership_evidence"] = {
            "real_etcd_lease_expiry_required": True,
            "n3_absent_observed_after_kill": self.metrics.saw_n3_absent_after_kill
                if self.metrics else False,
            "n3_return_observed": self.metrics.saw_n3_return if self.metrics else False,
        }
        self.result["churn"] = {
            "removers": [{"client": remover.client.client_id,
                          "start_second": remover.start_second,
                          "removed": remover.removed,
                          "rewritten": remover.rewritten,
                          "pending_rewrite_not_run_after_end": sum(
                              map(len, remover.pending_rewrite.values()))}
                         for remover in self.removers]
        }
        assert self.metrics is not None
        client_metrics = {client_id: _client_core(text)
                          for client_id, text in self.metrics.last_client_raw.items()}
        server_metrics = []
        for node in self.all_node_generations():
            text = self.metrics.last_server_raw.get(node.key, "")
            server_metrics.append({
                "node": node.node_id, "generation": node.generation,
                "key": node.key, "binary_role": node.binary_role,
                "binary": str(node.binary), "data_dir": str(node.data_dir),
                "metrics_available": bool(text), **_server_core(text),
            })
        readback = {
            label: sum(value or 0 for value in (
                metrics.get(f"readback_{label}") for metrics in client_metrics.values()))
            for label in READBACK_RESULTS
        }
        self.result["metrics"] = {"clients": client_metrics,
                                  "client_readback_totals": readback,
                                  "servers": server_metrics,
                                  "final_capture_complete": self.metrics.final_complete,
                                  "every_generation_has_metrics": all(
                                      row["metrics_available"] for row in server_metrics)}
        self._correctness = check_operations(self.root / "operations.jsonl")
        self.result["correctness"] = self._correctness
        self.result["metrics_comparison"] = {
            "readback_hit_rewritten": readback["hit_rewritten"],
            "readback_refused_removed": readback["refused_removed"],
            "server_removed_within_ages_requests": sum(
                row["removed_within_ages_requests"] or 0 for row in server_metrics),
            "server_removed_within_queries": sum(
                row["removed_within_queries"] or 0 for row in server_metrics),
        }
        self.result["evidence"] = {
            "operations": "operations.jsonl", "per_second": "per_second.jsonl",
            "prefix_results": "prefix_results.jsonl", "timeline": "timeline.jsonl",
            "metrics": "metrics.jsonl and metrics/*.prom",
            "membership": "membership.jsonl", "processes": "processes.jsonl",
            "logs": "logs/ (retained, including killed n3)",
            "killed_n3_data": self.kill_evidence.get("retained_data_dir"),
            "returned_n3_data": self.restart_evidence.get("fresh_empty_data_dir"),
        }

    def cleanup(self) -> list[str]:
        if self._cleaned_up:
            return []
        self._cleaned_up = True
        errors = []
        for remover in self.removers:
            remover.stop_event.set()
        try:
            self.join_readers()
        except Exception as exc:
            errors.append(f"reader cleanup: {exc}")
        for remover in self.removers:
            try:
                remover.stop_and_join()
            except Exception as exc:
                errors.append(f"remover cleanup: {exc}")
        if self.metrics is not None:
            try:
                self.metrics.stop_and_join()
            except Exception as exc:
                errors.append(f"metrics cleanup: {exc}")
            if not self._final_captured:
                try:
                    self.metrics.capture("final")
                    self._final_captured = True
                except Exception as exc:
                    errors.append(f"final metrics: {exc}")
        for client in self.client_snapshot():
            try:
                client.close()
            except Exception as exc:
                errors.append(f"client {client.client_id} close: {exc}")
            finally:
                with self.clients_lock:
                    self.clients.pop(client.client_id, None)
        if self.loader is not None:
            try:
                self.loader.close()
            except Exception as exc:
                errors.append(f"loader close: {exc}")
            self.loader = None
        self.restore_client_env()
        try:
            self.processes.close()
        except Exception as exc:
            errors.append(f"process cleanup: {exc}")
        live = [child.name for child in self.processes.children
                if child.popen.poll() is None]
        if live:
            errors.append(f"processes still alive: {live}")
        live_threads = [thread.name for thread in threading.enumerate()
                        if thread.name.startswith(("case5-reader", "remover-", "metrics"))
                        and thread.is_alive() and thread is not threading.current_thread()]
        if live_threads:
            errors.append(f"threads still alive: {live_threads}")
        for sink in self.sinks.values():
            try:
                sink.close()
            except Exception as exc:
                errors.append(f"evidence close: {exc}")
        self.result["cleanup"] = {
            "native_handles_open": len(self.clients),
            "processes_alive": live, "worker_threads_alive": live_threads,
            "errors": errors,
        }
        return errors

    def run(self) -> dict:
        try:
            self.execute()
        except BaseException as exc:
            self.result["errors"].append(f"{type(exc).__name__}: {exc}")
            (self.root / "error.log").write_text(traceback.format_exc())
        cleanup_errors = self.cleanup()
        self.result["errors"].extend(cleanup_errors)
        self.result["status"] = "ERROR" if self.result["errors"] else "COMPLETE"
        _write_json(self.root / "result.json", self.result)
        return self.result


def check_operations(path: Path) -> dict:
    """Apply the offline rule after all callers have joined.

    Only phase=workload PUTs can be witnesses; initial load and any future
    cleanup scope cannot hide a violation.  An unknown-end workload PUT is
    conservatively treated as ending at +infinity.  Such incomplete evidence
    is separately reported and prevents acceptance.
    """
    opens: dict[int, list[tuple[str, bytes, str, int, int | None]]] = {}
    history: dict[bytes, dict[str, list]] = {}
    events = 0
    for raw in path.open(encoding="utf-8"):
        if not raw:
            continue
        event = json.loads(raw)
        events += 1
        key = event["key"].encode()
        attempt = int(event["attempt"])
        op = event["op"]
        if event["event"] == "start":
            opens.setdefault(attempt, []).append(
                (op, key, event["phase"], int(event["start_ns"]), event.get("second")))
            continue
        if event["event"] != "end" or attempt not in opens:
            continue
        start_op, start_key, phase, start_ns, second = opens[attempt].pop(0)
        if not opens[attempt]:
            del opens[attempt]
        if start_op != op or start_key != key:
            raise RuntimeError(f"operation {attempt} start/end identity mismatch")
        item = history.setdefault(key, {"GET": [], "PUT": [], "REMOVE": []})
        if op in item and phase == "workload":
            item[op].append({
                "attempt": attempt, "start_ns": start_ns,
                "end_ns": int(event["end_ns"]), "second": second,
                "hit": bool(event.get("hit", False)),
                "payload_correct": event.get("payload_correct"),
            })
    incomplete = [{"attempt": attempt, "op": op, "key": key.decode(), "phase": phase}
                  for attempt, entries in opens.items()
                  for op, key, phase, _, _ in entries]
    for attempt, entries in opens.items():
        for op, key, phase, start_ns, second in entries:
            if op == "PUT" and phase == "workload":
                history.setdefault(key, {"GET": [], "PUT": [], "REMOVE": []})["PUT"].append({
                    "attempt": attempt, "start_ns": start_ns, "end_ns": None,
                    "second": second, "unknown_end_conservative": True,
                })
    violations = []
    checked_hits = 0
    for key, item in history.items():
        removes = sorted(item["REMOVE"], key=lambda row: row["start_ns"])
        puts = item["PUT"]
        for read in item["GET"]:
            if read["hit"] and read["payload_correct"] is False:
                violations.append({"type": "payload_mismatch", "key": key.decode(),
                                   "read": read})
            if not read["hit"]:
                continue
            checked_hits += 1
            candidates = [row for row in removes if row["end_ns"] < read["start_ns"]]
            if not candidates:
                continue
            remove = max(candidates, key=lambda row: row["start_ns"])
            witness = next((put for put in puts
                            if and_condition(put, remove, read)), None)
            if witness is None:
                violations.append({"type": "removed_value_hit", "key": key.decode(),
                                   "remove": remove, "read": read})
    return {
        "events": events, "checked_hit_reads": checked_hits,
        "violations": violations, "violation_count": len(violations),
        "incomplete_attempts": incomplete, "incomplete_attempt_count": len(incomplete),
        "rule": "latest REMOVE by start with end<read.start; violation iff no workload PUT has end>REMOVE.start and start<read.end; unknown PUT end=+infinity",
        "initial_and_cleanup_excluded": True,
        "evaluated_after_all_workers_joined": True,
    }


def and_condition(put: dict, remove: dict, read: dict) -> bool:
    # Separate helper keeps the unknown-end rule explicit and auditable.
    end_ok = put["end_ns"] is None or put["end_ns"] > remove["start_ns"]
    return end_ok and put["start_ns"] < read["end_ns"]


def add_check(result: dict, name: str, passed: bool, detail: Any) -> None:
    result.setdefault("acceptance_checks", []).append(
        {"name": name, "passed": bool(passed), "detail": detail})


def selected_checks(result: dict, smoke: bool) -> None:
    variant = result["variant"]
    correctness = result.get("correctness", {})
    add_check(result, "zero_correctness_violations",
              correctness.get("violation_count") == 0,
              {"count": correctness.get("violation_count"),
               "first": correctness.get("violations", [])[:3]})
    add_check(result, "complete_operation_intervals",
              correctness.get("incomplete_attempt_count") == 0,
              correctness.get("incomplete_attempts", [])[:3])
    coverage_windows = [result.get("core", {}).get("whole_run", {})] if smoke else list(
        result.get("core", {}).values())
    full_coverage = all(window.get("complete") and not window.get("errors")
                        for window in coverage_windows)
    add_check(result, "complete_per_second_workload", full_coverage,
              result.get("core", {}))
    restart = result.get("restart_evidence", {})
    kill = result.get("kill_evidence", {})
    add_check(result, "only_n3_killed_and_empty_fresh_return",
              kill.get("node") == "n3" and kill.get("returncode") == -9
              and restart.get("node") == "n3"
              and restart.get("fresh_directory_distinct") is True,
              {"kill": kill, "restart": restart})
    if not smoke:
        member = result.get("membership_evidence", {})
        add_check(result, "real_etcd_absence_and_return_observed",
                  member.get("n3_absent_observed_after_kill") is True
                  and member.get("n3_return_observed") is True, member)
    metrics = result.get("metrics", {})
    add_check(result, "authoritative_final_and_killed_generation_metrics",
              metrics.get("final_capture_complete") is True
              and metrics.get("every_generation_has_metrics") is True,
              {"final_capture_complete": metrics.get("final_capture_complete"),
               "every_generation_has_metrics": metrics.get("every_generation_has_metrics")})
    totals = metrics.get("client_readback_totals", {})
    servers = metrics.get("servers", [])
    ages = sum(row.get("removed_within_ages_requests") or 0 for row in servers)
    queries = sum(row.get("removed_within_queries") or 0 for row in servers)
    rewritten = totals.get("hit_rewritten", 0)
    refused = totals.get("refused_removed", 0) + totals.get("refused_unknown", 0)
    if smoke:
        return
    if variant == "base":
        add_check(result, "base_sends_no_M_or_W_queries", ages == 0 and queries == 0,
                  {"ages_requests": ages, "removed_within_queries": queries})
    if variant in ("amb", "ambw-woff"):
        add_check(result, "client_rewrite_off_means_no_op26", ages == 0,
                  {"ages_requests": ages})
        add_check(result, "B_readback_path_exercised", queries > 0,
                  {"removed_within_queries": queries})
        add_check(result, "no_rewritten_hits_without_client_rewrite", rewritten == 0,
                  {"hit_rewritten": rewritten})
    if variant == "ambw-woff":
        write_entries = sum(row.get("write_memory_entries") or 0 for row in servers)
        add_check(result, "server_W_active_while_client_rewrite_off", write_entries > 0,
                  {"write_memory_entries": write_entries})
    if variant == "ambw-now":
        add_check(result, "W_off_servers_receive_op26_and_fail_closed",
                  ages > 0 and rewritten == 0 and refused > 0,
                  {"ages_requests": ages, "hit_rewritten": rewritten,
                   "refused_removed_or_unknown": refused,
                   "reason": "all current servers have DFKV_WRITE_MEMORY_ENTRIES=0"})
    if variant == "ambw-old":
        old = [row for row in servers if row.get("binary_role") == "old"]
        current = [row for row in servers if row.get("binary_role") == "current"]
        old_queries = sum(row.get("removed_within_queries") or 0 for row in old)
        old_ages = sum(row.get("removed_within_ages_requests") or 0 for row in old)
        current_ages = sum(row.get("removed_within_ages_requests") or 0 for row in current)
        add_check(result, "real_op25_only_old_server_fallback",
                  bool(old) and old_queries > 0 and old_ages == 0 and current_ages > 0
                  and refused > 0,
                  {"old_servers": old, "old_op25_keys": old_queries,
                   "old_op26_requests": old_ages, "current_op26_requests": current_ages,
                   "refused_removed_or_unknown": refused})


def cross_variant_checks(by_name: dict[str, dict]) -> dict:
    required = {"base", "amb", "ambw"}
    if not required.issubset(by_name):
        return {"evaluated": False,
                "missing": sorted(required.difference(by_name))}
    base = by_name["base"]["core"]["post_return_30_60s"]
    amb = by_name["amb"]
    ambw = by_name["ambw"]
    first = ambw["core"]["first_10s_post_return"]
    if not base.get("complete") or not first.get("complete") or not base.get("prefixes") or not first.get("prefixes"):
        add_check(ambw, "ambw_first10_at_least_90pct_base_steady", False,
                  "incomplete core-rate windows")
        return {"evaluated": False, "reason": "incomplete core-rate windows"}
    base_rate = Fraction(base["full_prefixes"], base["prefixes"])
    first_rate = Fraction(first["full_prefixes"], first["prefixes"])
    ratio = first_rate / base_rate if base_rate else None
    passes = base_rate > 0 and first_rate >= Fraction(9, 10) * base_rate
    add_check(ambw, "ambw_first10_at_least_90pct_base_steady", passes, {
        "ambw_first10": first, "base_30_60_post_return": base,
        "ratio_exact": str(ratio) if ratio is not None else None,
        "threshold_exact": "9/10",
    })
    amb_refused = int(amb["metrics"]["client_readback_totals"]["refused_removed"])
    ambw_refused = int(ambw["metrics"]["client_readback_totals"]["refused_removed"])
    reduction = (Fraction(amb_refused - ambw_refused, amb_refused)
                 if amb_refused else None)
    add_check(ambw, "refused_removed_reduced_by_more_than_half_vs_amb",
              amb_refused > 0 and ambw_refused < amb_refused * 0.5, {
                  "amb_refused_removed": amb_refused,
                  "ambw_refused_removed": ambw_refused,
                  "reduction_exact": str(reduction) if reduction is not None else None,
                  "threshold": "strictly greater than 1/2",
              })
    rewritten = int(ambw["metrics"]["client_readback_totals"]["hit_rewritten"])
    add_check(ambw, "nonzero_hit_rewritten", rewritten > 0,
              {"hit_rewritten": rewritten})
    amb_first30 = amb["core"]["first_30s_post_return"]
    amb_first10 = amb["core"]["first_10s_post_return"]
    def exact_ratio(window: dict) -> str | None:
        if not window.get("prefixes") or not base_rate:
            return None
        return str(Fraction(window["full_prefixes"], window["prefixes"]) / base_rate)
    return {
        "evaluated": True,
        "base_post_return_30_60s_rate_exact": str(base_rate),
        "amb_first_10s_over_base_exact": exact_ratio(amb_first10),
        "amb_first_30s_over_base_exact": exact_ratio(amb_first30),
        "ambw_first_10s_over_base_exact": str(ratio) if ratio is not None else None,
        "amb_refused_removed": amb_refused,
        "ambw_refused_removed": ambw_refused,
        "refused_removed_reduction_exact": str(reduction) if reduction is not None else None,
        "ambw_hit_rewritten": rewritten,
    }


def load_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.dfkv_open_v2.restype = ctypes.c_void_p
    lib.dfkv_open_v2.argtypes = [ctypes.POINTER(DfkvClientOptionsV2)]
    for name in ("dfkv_batch_get_auto_st", "dfkv_batch_put_st"):
        if not hasattr(lib, name):
            raise RuntimeError(f"{path} lacks required {name}")
    lib.dfkv_batch_get_auto_st.restype = ctypes.c_int
    lib.dfkv_batch_get_auto_st.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.dfkv_batch_put_st.restype = ctypes.c_int
    lib.dfkv_batch_put_st.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint8),
    ]
    for name in ("dfkv_batch_remove", "dfkv_batch_exist"):
        function = getattr(lib, name)
        function.restype = ctypes.c_int
        function.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
    lib.dfkv_transport_mode.restype = ctypes.c_char_p
    lib.dfkv_transport_mode.argtypes = [ctypes.c_void_p]
    lib.dfkv_stats_snapshot.restype = ctypes.c_uint64
    lib.dfkv_stats_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64]
    lib.dfkv_close.restype = None
    lib.dfkv_close.argtypes = [ctypes.c_void_p]
    hasattr(lib, "dfkv_version")
    lib.dfkv_version.restype = ctypes.c_char_p
    lib.dfkv_version.argtypes = []
    return lib


def discover_etcd() -> Path:
    configured = os.environ.get("DFKV_ETCD") or os.environ.get("ETCD_BIN")
    resolved = shutil.which(configured) if configured else shutil.which("etcd")
    if not resolved:
        raise RuntimeError("real etcd not found: set DFKV_ETCD/ETCD_BIN or add etcd to PATH")
    path = Path(resolved).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"etcd is not executable: {path}")
    return path


def executable(path: Path, what: str) -> Path:
    path = path.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"{what} is not an executable file: {path}")
    return path


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--old-server", type=Path,
                        help="real op25-only dfkv_server for ambw-old")
    parser.add_argument("--variants", help=",".join(ALL_VARIANTS) + " (CSV debug subset)")
    parser.add_argument("--smoke", action="store_true",
                        help="shorten only timing; mechanics evidence is non-acceptance")
    args = parser.parse_args(argv)
    variants = tuple(args.variants.split(",")) if args.variants else ALL_VARIANTS
    if not variants or any(v not in ALL_VARIANTS for v in variants):
        parser.error(f"--variants must be a CSV of {','.join(ALL_VARIANTS)}")
    if len(set(variants)) != len(variants):
        parser.error("duplicate --variants entries are not allowed")
    args.variants = variants
    build = args.build_dir.resolve()
    args.server = executable(build / "dfkv_server", "dfkv_server")
    args.mds = executable(build / "dfkv_mds", "dfkv_mds")
    args.ctl = executable(build / "dfkvctl", "dfkvctl")
    args.lib = build / "libdfkv.so"
    if not args.lib.is_file():
        parser.error(f"missing libdfkv.so: {args.lib}")
    args.lib = args.lib.resolve()
    if "ambw-old" in variants:
        if args.old_server is None:
            parser.error("ambw-old needs --old-server PATH (also required by the default full matrix)")
        args.old_server = executable(args.old_server, "op25-only old server")
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()):
        parser.error(f"output root must be empty so evidence is never overwritten: {args.output_root}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    etcd = discover_etcd()
    args.lib_handle = load_library(args.lib)
    version = args.lib_handle.dfkv_version()
    workload = Workload()
    timing = ({"kill_s": 6, "restart_s": 11, "end_s": 16,
               "down_s": 5, "post_return_s": 5, "second_remover_s": 8}
              if args.smoke else
              {"kill_s": 60, "restart_s": 150, "end_s": 240,
               "down_s": 90, "post_return_s": 90, "second_remover_s": 90})
    metadata = {
        "wall_time": _iso(), "argv": sys.argv if argv is None else [sys.argv[0], *argv],
        "cwd": os.getcwd(), "python": sys.version, "platform": sys.platform,
        "build_dir": str(args.build_dir.resolve()), "output_root": str(args.output_root),
        "requested_variants": args.variants, "full_matrix": set(args.variants) == set(ALL_VARIANTS),
        "smoke": args.smoke, "timing": timing,
        "binaries": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in (("server", args.server), ("mds", args.mds),
                              ("dfkvctl", args.ctl), ("libdfkv", args.lib), ("etcd", etcd),
                              *((("old_server", args.old_server),) if args.old_server else ()))
        },
        "libdfkv_version": version.decode(errors="replace") if version else "",
        "environment_whitelist": {key: os.environ.get(key) for key in CLIENT_ENV_KEYS},
        "etcd_discovery": "DFKV_ETCD, ETCD_BIN, then PATH; real process and /health required",
        "assumptions": [
            "Linux loopback TCP; no RDMA or GPU; no production endpoints",
            "enough disk below output-root for retained data/logs and 32 MiB initial values per variant",
            "old server is wire/MDS compatible, implements op25, and predates op26",
            "C is off because case5 specifies A(+M+B(+W)), not cooldown reroute",
        ],
        "workload_identity_sha256": workload.identity_sha256,
    }
    _write_json(args.output_root / "run-metadata.json", metadata)
    results = []
    for variant in args.variants:
        print(f"[case5] starting {variant}; evidence={args.output_root / variant}", flush=True)
        result = VariantRun(args, variant, timing, workload, etcd).run()
        results.append(result)
        core = result.get("core", {})
        first = core.get("first_10s_post_return", {})
        steady = core.get("post_return_30_60s", {})
        totals = result.get("metrics", {}).get("client_readback_totals", {})
        print(
            f"[case5:{variant}] status={result['status']} "
            f"first10={first.get('full_prefix_hit_rate')} "
            f"post30-60={steady.get('full_prefix_hit_rate')} "
            f"hit_rewritten={totals.get('hit_rewritten', 0)} "
            f"refused_removed={totals.get('refused_removed', 0)} "
            f"violations={result.get('correctness', {}).get('violation_count', 'unknown')}",
            flush=True,
        )
        if result["status"] == "ERROR":
            print(f"[case5:{variant}] {result['errors']}", file=sys.stderr, flush=True)
            break
    by_name = {result["variant"]: result for result in results}
    comparisons: dict = {"evaluated": False}
    for result in results:
        if result["status"] == "ERROR":
            continue
        selected_checks(result, args.smoke)
        failures = [check for check in result["acceptance_checks"] if not check["passed"]]
        if failures:
            result["status"] = "FAILED"
        elif args.smoke:
            result["status"] = "SMOKE_MECHANICS_COMPLETE_NON_ACCEPTANCE"
        else:
            result["status"] = "CHECKED"
    if not args.smoke and all(result["status"] != "ERROR" for result in results):
        comparisons = cross_variant_checks(by_name)
        for name in ("base", "amb", "ambw"):
            if name in by_name:
                # Cross checks appended to ambw above are part of that variant's result.
                failures = [check for check in by_name[name].get("acceptance_checks", [])
                            if not check["passed"]]
                if failures:
                    by_name[name]["status"] = "FAILED"
    full_matrix = set(args.variants) == set(ALL_VARIANTS)
    all_ran = len(results) == len(args.variants)
    failed = any(result["status"] in ("FAILED", "ERROR") for result in results)
    if args.smoke:
        overall = "SMOKE_FAILED" if failed else "SMOKE_MECHANICS_COMPLETE_NON_ACCEPTANCE"
    elif not full_matrix or not all_ran:
        overall = "FAILED" if failed else "DEBUG_SUBSET_COMPLETE_NON_ACCEPTANCE"
    elif failed:
        overall = "ACCEPTANCE_FAIL"
    else:
        overall = "ACCEPTANCE_PASS"
    summary = {
        "wall_time": _iso(), "overall": overall, "smoke": args.smoke,
        "full_matrix": full_matrix, "all_requested_variants_ran": all_ran,
        "acceptance": overall == "ACCEPTANCE_PASS",
        "non_acceptance_reason": (
            "--smoke validates mechanics only" if args.smoke else
            "explicit --variants subset is debugging, not full acceptance" if not full_matrix else None),
        "comparison": comparisons, "variants": results,
    }
    _write_json(args.output_root / "result.json", summary)
    for result in results:
        _write_json(args.output_root / result["variant"] / "result.json", result)
    print(f"[case5] comparison={json.dumps(comparisons, sort_keys=True)}", flush=True)
    if overall == "ACCEPTANCE_PASS":
        print("[case5] ACCEPTANCE PASS", flush=True)
    elif overall.startswith("SMOKE"):
        print(f"[case5] {overall}; no acceptance PASS claimed", flush=True)
    else:
        print(f"[case5] {overall}; result={args.output_root / 'result.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
