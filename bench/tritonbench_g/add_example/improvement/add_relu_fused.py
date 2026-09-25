import torch
import triton
import triton.language as tl

@triton.jit
def add_relu_kernel(
    in_ptr0: tl.tensor,
    in_ptr1: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    z = x + y
    output = tl.where(z > 0, z, 0)
    tl.store(out_ptr + offsets, output, mask=mask)

def add_relu_wrapper(x, y):
    out = torch.empty_like(x)  # every element is written (AddReluFused.lean: coverage conjunct)
    
    BLOCK_SIZE = 64
    n_elements = x.numel()

    # Calculate the number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    add_relu_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)

    return out




##################################################################################################################################################
# Candidate (not a corpus port): fused float32 add + ReLU, relu(x + y), with add_example's launch arithmetic.
# The ReLU is spelled as relu_strided_buffer's relu_forward (tl.where(x > 0, x, 0)).
