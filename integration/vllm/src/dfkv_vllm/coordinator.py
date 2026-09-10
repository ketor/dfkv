# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-group persistence and restoration masks for DfkvStoreConnector."""

from typing import cast

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

# Dummy placeholder hash for store_mask's template computation.
_DUMMY_BLOCK_HASH = BlockHash(b"\x00" * 32)


def _unwrap_hit_blocks(res):
    """vLLM >= 0.26: SingleTypeKVCacheManager.find_longest_cache_hit returns
    ``(blocks_per_group, hit_length_tokens)``; older versions return just
    ``blocks_per_group``. Accept both. hit_length is always re-derived from
    the returned block lists, so the manager-side value is not needed here.
    """
    if (
        isinstance(res, tuple)
        and len(res) == 2
        and isinstance(res[0], tuple)
        and isinstance(res[1], int)
    ):
        return res[0]
    return res


class _TemplateBlockPool:
    """All-present keys for deriving a manager's required state window."""

    def __init__(self, hash_block_size: int) -> None:
        self.hash_block_size = hash_block_size
        self.null_block = KVCacheBlock(block_id=0)
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock]:
        return [self._present_block] * len(group_ids)


def _prefix_cacheable(spec):
    """Older engines name this participates_in_prefix_caching."""
    for name in ("prefix_cacheable", "participates_in_prefix_caching"):
        value = getattr(spec, name, None)
        if value is not None:
            return bool(value)
    return True


class DfkvStoreCoordinator:
    """Build per-group cache masks using the running engine's managers."""
    def __init__(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
        use_eagle: bool = False,
    ) -> None:

        participating = [
            (_prefix_cacheable(g.kv_cache_spec), g.kv_cache_spec.block_size)
            for g in kv_cache_groups
        ]
        assert all(
            size % hash_block_size == 0
            for participates, size in participating
            if participates
        ), (
            f"block_size must be divisible by hash_block_size "
            f"(hash={hash_block_size}, scheduler={scheduler_block_size}, "
            f"groups={participating})"
        )
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of "
            f"hash_block_size ({hash_block_size})"
        )
        assert all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_groups
        ), "scheduler_block_size must be a multiple of each group's block_size"
        self.kv_cache_groups = kv_cache_groups
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.use_eagle = use_eagle
        self._verify_and_split_kv_cache_groups()

    def _verify_and_split_kv_cache_groups(self) -> None:
        """Group equal specs and resolve their engine-owned managers."""
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []
        for i, g in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(g.kv_cache_spec)
            if not _prefix_cacheable(g.kv_cache_spec):
                continue
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None, (
                f"No manager registered for KVCacheSpec {spec}"
            )
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))
        self.attention_groups = attention_groups
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if self.use_eagle
            and any(self.kv_cache_groups[gid].is_eagle_group for gid in group_ids)
        }
        if self.use_eagle and not self.eagle_attn_group_indices:
            self.eagle_attn_group_indices = set(range(len(self.attention_groups)))


    def load_mask(
        self,
        block_hashes: list[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...]:
        """Restore every group required at the already-admitted hit boundary."""
        aligned = token_len // self.lcm_block_size * self.lcm_block_size
        return self._cache_mask(aligned, drop_eagle=False)

    def store_mask(self, aligned_token_len: int) -> tuple[list[bool], ...]:
        """Persist only stable context, never a volatile draft/lookahead tail."""
        return self._cache_mask(aligned_token_len, drop_eagle=True)

    def _cache_mask(
        self,
        aligned_token_len: int,
        *,
        drop_eagle: bool,
    ) -> tuple[list[bool], ...]:
        assert aligned_token_len % self.lcm_block_size == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of "
            f"lcm_block_size ({self.lcm_block_size})"
        )
        if aligned_token_len == 0:
            return tuple([] for _ in self.kv_cache_groups)

        dummy_hashes: list[BlockHash] = [_DUMMY_BLOCK_HASH] * (
            aligned_token_len // self.hash_block_size
        )
        block_pool = _TemplateBlockPool(hash_block_size=self.hash_block_size)
        masks: list[list[bool]] = [[] for _ in self.kv_cache_groups]
        for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
            hashes = self.block_hashes_for_spec(dummy_hashes, spec)
            hit_blocks = _unwrap_hit_blocks(
                manager_cls.find_longest_cache_hit(
                    block_hashes=hashes,
                    max_length=aligned_token_len,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, block_pool),
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle and idx in self.eagle_attn_group_indices,
                    alignment_tokens=self.lcm_block_size,
                )
            )
            for group_id, blocks in zip(group_ids, hit_blocks, strict=True):
                masks[group_id] = [
                    block is not block_pool.null_block for block in blocks
                ]
        return tuple(masks)

    def block_hashes_for_spec(
        self, block_hashes: list[BlockHash], spec: KVCacheSpec
    ) -> BlockHashList:
        if spec.block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes, self.hash_block_size, spec.block_size
        )



def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec
