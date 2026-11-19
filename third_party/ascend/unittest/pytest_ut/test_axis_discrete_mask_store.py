"""Axis-discrete mask store on the A5/A6 register pipeline.

The store carve-out lowers to a read-modify-write: load the tile, select the
masked lanes, write it back. Positions where the mask is false must keep their
original value.
"""

import re

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_register_based_arch


def _skip_unless_register_based():
    arch = triton.runtime.driver.active.get_current_target().arch
    if not is_register_based_arch(arch):
        pytest.skip(f"axis-discrete SIMD routing requires A5/A6, current arch is {arch}")


def _adapter(compiled, kernel_name):
    assert "ttadapter" in compiled.asm, f"{kernel_name}: no ttadapter stage was produced"
    return compiled.asm["ttadapter"]


def _assert_simd_store_routing(compiled, kernel_name):
    """The store must stay on the vectorized read-modify-write path."""
    adapter = _adapter(compiled, kernel_name)
    assert "indirect_store" not in adapter, (
        f"{kernel_name}: masked store fell back to the SIMT template path")
    assert "indirect_load" not in adapter, (
        f"{kernel_name}: read-modify-write load fell back to the SIMT template path")
    assert "arith.select" in adapter, "masked lanes were not folded into arith.select"
    return adapter


@triton.jit
def _axis_discrete_mask_store_kernel(
    out_ptr,
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    k = tl.arange(0, BLOCK_K)
    g = tl.arange(0, G)

    # Pure discrete mask: varies along rows, constant along the contiguous axis.
    row_keep = (k % 2) == 0
    val = tl.full([BLOCK_K, G], 7.0, tl.float16)
    tl.store(out_ptr + k[:, None] * G + g[None, :], val, mask=row_keep[:, None])


@triton.jit
def _axis_discrete_mask_store_and_kernel(
    out_ptr,
    limit,  # runtime scalar column bound
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    k = tl.arange(0, BLOCK_K)
    g = tl.arange(0, G)

    row_keep = (k % 2) == 0
    col_keep = g < limit
    mask = row_keep[:, None] & col_keep[None, :]
    val = tl.full([BLOCK_K, G], 7.0, tl.float16)
    tl.store(out_ptr + k[:, None] * G + g[None, :], val, mask=mask)


@triton.jit
def _axis_c1_mask_store_kernel(
    out_ptr,
    bounds_ptr,
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    k = tl.arange(0, BLOCK_K)
    g = tl.arange(0, G)

    # C1: each row has its own runtime length; no axis is constant.
    bound = tl.load(bounds_ptr + k)
    mask = g[None, :] < bound[:, None]
    val = tl.full([BLOCK_K, G], 7.0, tl.float16)
    tl.store(out_ptr + k[:, None] * G + g[None, :], val, mask=mask)


def _init_out(BLOCK_K, G):
    return (torch.arange(BLOCK_K * G, dtype=torch.float16, device="npu") % 101).reshape(
        BLOCK_K, G)


@pytest.mark.parametrize("G,BLOCK_K", [
    (128, 64),
    (64, 128),
])
def test_axis_discrete_mask_store(G, BLOCK_K):
    _skip_unless_register_based()
    out = _init_out(BLOCK_K, G)
    original = out.clone()

    compiled = _axis_discrete_mask_store_kernel[(1, )](out, BLOCK_K=BLOCK_K, G=G)
    torch.npu.synchronize()

    k = torch.arange(BLOCK_K, device="npu")[:, None]
    expected = torch.where((k % 2) == 0, torch.full_like(original, 7.0), original)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    adapter = _assert_simd_store_routing(compiled, "axis_discrete_mask_store")
    assert "arith.select" in adapter


@pytest.mark.parametrize("G,BLOCK_K,limit", [
    (128, 64, 96),
    (64, 128, 32),
])
def test_axis_discrete_mask_store_and(G, BLOCK_K, limit):
    _skip_unless_register_based()
    out = _init_out(BLOCK_K, G)
    original = out.clone()

    compiled = _axis_discrete_mask_store_and_kernel[(1, )](
        out, limit, BLOCK_K=BLOCK_K, G=G)
    torch.npu.synchronize()

    k = torch.arange(BLOCK_K, device="npu")[:, None]
    g = torch.arange(G, device="npu")[None, :]
    keep = ((k % 2) == 0) & (g < limit)
    expected = torch.where(keep, torch.full_like(original, 7.0), original)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    adapter = _assert_simd_store_routing(compiled, "axis_discrete_mask_store_and")
    # contMask must bound the read-modify-write; without it the whole tile moves.
    assert re.search(r"memref\.subview[^\n]*\[\d+, %\w+\]", adapter), \
        "contMask did not narrow the read-modify-write range"


@pytest.mark.parametrize("G,BLOCK_K", [
    (64, 32),
    (128, 64),
])
def test_axis_c1_mask_store_falls_back_to_simt(G, BLOCK_K):
    _skip_unless_register_based()
    out = _init_out(BLOCK_K, G)
    original = out.clone()
    bounds = torch.randint(0, G + 1, (BLOCK_K, ), dtype=torch.int32, device="npu")

    compiled = _axis_c1_mask_store_kernel[(1, )](
        out, bounds, BLOCK_K=BLOCK_K, G=G)
    torch.npu.synchronize()

    g = torch.arange(G, device="npu")[None, :]
    keep = g < bounds[:, None]
    expected = torch.where(keep, torch.full_like(original, 7.0), original)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    adapter = _adapter(compiled, "axis_c1_mask_store")
    assert "indirect_store" in adapter, (
        "per-row variable-length mask unexpectedly avoided the SIMT fallback")


@triton.jit
def _multi_program_axis_discrete_store_kernel(
    out_ptr,
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    g = tl.arange(0, G)

    # Two programs write disjoint rows of the same tile.  Each store is a
    # full-tile read-modify-write, so without the block lock the later writer
    # can put back stale values for the other program's rows.
    keep = (k % 2) == pid
    val = tl.full([BLOCK_K, G], 0.0, tl.float16) + (pid + 1).to(tl.float16)
    tl.store(out_ptr + k[:, None] * G + g[None, :], val, mask=keep[:, None])


@pytest.mark.parametrize("G,BLOCK_K", [
    (128, 64),
    (64, 32),
])
def test_multi_program_store_falls_back_to_simt(G, BLOCK_K):
    """Overlapping read-modify-write windows must use the SIMT per-lane store.

    With sync_block_lock the two full-tile passes serialize and measure ~2x the
    SIMT fallback, so the carve-out is deliberately disabled for this case.
    """
    _skip_unless_register_based()

    for _ in range(5):
        out = torch.zeros((BLOCK_K, G), dtype=torch.float16, device="npu")
        compiled = _multi_program_axis_discrete_store_kernel[(2, )](
            out, BLOCK_K=BLOCK_K, G=G)
        torch.npu.synchronize()

        k = torch.arange(BLOCK_K, device="npu")[:, None]
        expected = torch.where((k % 2) == 0,
                               torch.full_like(out, 1.0),
                               torch.full_like(out, 2.0))
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    adapter = _adapter(compiled, "multi_program_axis_discrete_store")
    assert "indirect_store" in adapter, (
        "overlapping stores should stay on the SIMT per-lane store path")
    assert "sync_block_lock" not in adapter, (
        "overlapping stores should not take the lock-serialized read-modify-write")

