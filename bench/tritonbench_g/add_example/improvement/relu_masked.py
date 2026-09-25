import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    in_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    output = tl.where(x > 0, x, 0)
    tl.store(out_ptr + offsets, output, mask=mask)




##################################################################################################################################################
# Candidate (not a corpus port): masked-pointer ReLU with add_example's launch arithmetic;
# the second launch of the two-launch pipeline in AddReluRelational.lean.
