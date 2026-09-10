from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dfkv_vllm.data import LoadSpec
from dfkv_vllm.scheduler import DfkvStoreScheduler


def test_new_request_metadata_uses_complete_allocated_block_table():
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler.client = MagicMock()
    scheduler.load_specs = {}
    scheduler._request_trackers = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = set()
    scheduler._allocated_req_ids = set()
    scheduler._block_size = 4

    request_real = SimpleNamespace(request_id="req-1", block_hashes=[])
    complete_block_ids = ([10, 11], [20, 21])
    scheduler._unfinished_requests = {}
    blocks = SimpleNamespace(get_block_ids=lambda: complete_block_ids)
    scheduler.update_state_after_alloc(request_real, blocks, num_external_tokens=0)

    scheduled_new_request = SimpleNamespace(
        req_id="req-1",
        num_computed_tokens=0,
        # vLLM's NewRequestData exposes only blocks allocated this step.
        block_ids=([11], [21]),
        prefill_token_ids=None,
        prompt_token_ids=list(range(8)),
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[scheduled_new_request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"req-1": 8},
    )

    metadata = scheduler.build_connector_meta(scheduler_output)

    assert len(metadata.requests) == 1
    assert metadata.requests[0].block_ids == complete_block_ids
    assert metadata.requests[0].can_save is True


def test_sampling_tail_does_not_authorize_an_absent_checkpoint():
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._block_size = 64
    scheduler.lookup_async = False
    scheduler.load_async = False
    scheduler.load_specs = {}
    checkpoints = {128}
    scheduler.client = SimpleNamespace(
        lookup=lambda request_id, length, hashes, non_block: max(
            (depth for depth in checkpoints if depth <= length), default=0
        )
    )
    request = SimpleNamespace(
        request_id="stateful-full-hit", num_tokens=128, block_hashes=[]
    )
    # The final token must be recomputed for sampling. A checkpoint at 128
    # cannot be used to resume at 64 when the latter checkpoint is absent.
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.request_id not in scheduler.load_specs
    checkpoints.add(64)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (64, False)


def test_cache_bypass_discards_stale_external_admission():
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._block_size = 64
    scheduler.lookup_async = False
    scheduler.load_async = False
    cached = {"replay": 128}
    other_spec = object()
    scheduler.load_specs = {"replay": object(), "unrelated": other_spec}
    scheduler.client = SimpleNamespace(
        lookup=lambda request_id, *args, **kwargs: cached[request_id],
        discard=lambda request_id: cached.pop(request_id, None),
    )
    request = SimpleNamespace(
        request_id="replay", num_tokens=256, block_hashes=[],
        skip_reading_prefix_cache=True,
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert cached == {}
    assert scheduler.load_specs == {"unrelated": other_spec}


@pytest.mark.parametrize("cached_resume", [False, True])
@pytest.mark.parametrize("async_pending", [False, True])
def test_same_step_preemption_keeps_new_allocation_and_load(async_pending, cached_resume):
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler.client = MagicMock()
    scheduler.load_specs = {}
    scheduler._request_trackers = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = set()
    scheduler._unfinished_requests = {}
    scheduler._allocated_req_ids = set()
    scheduler._block_size = 4
    request = SimpleNamespace(
        request_id="resumed", block_hashes=[], all_token_ids=list(range(16)),
        num_computed_tokens=4,
    )
    new_blocks = ([10, 11], [20, 21])
    scheduler.load_specs["resumed"] = LoadSpec(0, 4, False)
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: new_blocks), 4,
    )
    new_request = SimpleNamespace(
        req_id="resumed", num_computed_tokens=4, block_ids=([11], [21]),
        prefill_token_ids=list(range(16)), prompt_token_ids=list(range(8)),
    )
    resumed_cached = SimpleNamespace(
        req_ids=["resumed"], new_block_ids=[([11], [21])],
        num_computed_tokens=[4],
    )
    step = SimpleNamespace(
        finished_req_ids=set(), preempted_req_ids={"resumed"},
        scheduled_new_reqs=[] if async_pending or cached_resume else [new_request],
        scheduled_cached_reqs=(
            resumed_cached if cached_resume and not async_pending
            else SimpleNamespace(req_ids=[])
        ),
        num_scheduled_tokens={} if async_pending else {"resumed": 4},
    )
    metadata = scheduler.build_connector_meta(step)
    assert len(metadata.requests) == 1
    restored = metadata.requests[0]
    assert restored.load_spec is not None and restored.load_spec.can_load
    assert restored.block_ids == new_blocks

    if async_pending:
        # The load finishes later; either runner can now resume computation.
        if cached_resume:
            # The asynchronously restored table already has room for this step.
            resumed_cached.new_block_ids = [None]
        step.preempted_req_ids = set()
        step.scheduled_new_reqs = [] if cached_resume else [new_request]
        step.scheduled_cached_reqs = resumed_cached if cached_resume else SimpleNamespace(req_ids=[])
        step.num_scheduled_tokens = {"resumed": 4}
        scheduler.build_connector_meta(step)

    # The next chunk extends the full table; it is not another resume whose
    # delta-only allocation can replace the previously restored blocks.
    request.num_computed_tokens = 8
    step.preempted_req_ids = set()
    step.scheduled_new_reqs = []
    step.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["resumed"], new_block_ids=[([12], [22])],
        num_computed_tokens=[8],
    )
    step.num_scheduled_tokens = {"resumed": 4}
    metadata = scheduler.build_connector_meta(step)
    assert metadata.requests[0].block_ids == ([10, 11, 12], [20, 21, 22])

    # A later preemption without reallocation must not reuse the earlier
    # step's allocation marker or emit a load into recycled blocks.
    scheduler.load_specs["resumed"] = LoadSpec(0, 4, True)
    step.preempted_req_ids = {"resumed"}
    step.scheduled_cached_reqs = SimpleNamespace(req_ids=[])
    step.num_scheduled_tokens = {}
    metadata = scheduler.build_connector_meta(step)
    assert metadata.requests == []
