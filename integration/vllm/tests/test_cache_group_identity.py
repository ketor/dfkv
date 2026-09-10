import torch

from dfkv_common import canonical_namespace
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from dfkv_vllm.data import VLLM_RAW_LAYOUT
from dfkv_vllm.worker import _cache_group_layout


def _namespace(groups):
    return canonical_namespace(
        "DeepSeek-V4.1-Flash", VLLM_RAW_LAYOUT.decode("ascii"),
        model_revision="same-checkpoint", dtype="fp8", block_tokens=128,
        layout_fields={"group_layout": _cache_group_layout(groups)},
    )


def _indexer_spec(width):
    return MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=width,
        dtype=torch.uint8, tokens_per_state=2, alignment=512,
    )


def test_indexer_precision_changes_external_cache_identity():
    # Main KV dtype stays fp8; the indexer independently selects FP8 or MXFP4.
    fp8 = [KVCacheGroupSpec(["layer.indexer.k_cache"], _indexer_spec(132))]
    fp4 = [KVCacheGroupSpec(["layer.indexer.k_cache"], _indexer_spec(68))]
    assert _namespace(fp8) != _namespace(fp4)


def test_equal_sized_state_payloads_with_different_dtypes_are_isolated():
    first = MambaSpec(
        block_size=128, shapes=((32,),), dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    second = MambaSpec(
        block_size=128, shapes=((64,),), dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    assert first.page_size_bytes == second.page_size_bytes
    assert _namespace([KVCacheGroupSpec(["state"], first)]) != _namespace(
        [KVCacheGroupSpec(["state"], second)]
    )


def test_uniform_spec_mapping_order_does_not_change_external_identity():
    specs = {"layer.a": _indexer_spec(132), "layer.b": _indexer_spec(68)}
    first = KVCacheGroupSpec(
        list(specs), UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs)
    )
    reverse = dict(reversed(list(specs.items())))
    second = KVCacheGroupSpec(
        list(reverse),
        UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=reverse),
    )
    assert _namespace([first]) == _namespace([second])
