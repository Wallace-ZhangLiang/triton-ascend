# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common

# ---------------
# test tl.flip op
# ---------------
# `tl.flip` is lowered in the frontend as a bit-reversal idiom (reshape the
# 2^k-sized dim into k trailing 2s, xor-swap along each of them, then reshape
# back). The TritonToLinalg pass recognizes that idiom and rewrites it into the
# native `ascend.flip`. This test covers the flip itself: the flipped dim must
# be a power of two, and the result must exactly match torch.flip for any dtype
# (the xor-swap is a pure permutation, so even floats are bit-exact).


@triton.jit
def flip1d_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, dim: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.flip(x, dim=dim)
    tl.store(out_ptr + offs, y)


@triton.jit
def flip2d_kernel(x_ptr, out_ptr, ROWS: tl.constexpr, COLS: tl.constexpr, dim: tl.constexpr):
    r = tl.arange(0, ROWS)[:, None]
    c = tl.arange(0, COLS)[None, :]
    x = tl.load(x_ptr + r * COLS + c)
    y = tl.flip(x, dim=dim)
    tl.store(out_ptr + r * COLS + c, y)


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8'])
def test_flip_1d(dtype):
    size = 64
    x = test_common.generate_tensor((size, ), dtype).npu()
    torch_ref = torch.flip(x, dims=[0])
    triton_res = torch.empty_like(x)
    flip1d_kernel[(1, )](x, triton_res, size, 0)
    assert (torch_ref == triton_res).all(), (torch_ref, triton_res)


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8'])
@pytest.mark.parametrize("shape,dim", [
    ((16, 64), 1),   # flip the trailing 2^6 dim (issue #1638 repro)
    ((64, 64), 1),   # square matrix, trailing dim
    ((64, 16), 0),   # flip the leading 2^6 dim
    ((8, 32), 0),    # smaller 2^3 split on the leading dim
    ((4, 8), 1),     # smallest split this pattern rewrites (2^2 -> [2, 2])
])
def test_flip_2d(dtype, shape, dim):
    rows, cols = shape
    x = test_common.generate_tensor(shape, dtype).npu()
    torch_ref = torch.flip(x, dims=[dim])
    triton_res = torch.empty_like(x)
    flip2d_kernel[(1, )](x, triton_res, rows, cols, dim)
    assert (torch_ref == triton_res).all(), (torch_ref, triton_res)
