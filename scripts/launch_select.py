#!/usr/bin/env python3
"""The selected general checked API for the fused add + ReLU (stage 9).

`CheckedFusedAddReluSelected` = `launch_fast.CheckedFusedAddReluFast` with a
deterministic block rule: BLOCK_SIZE 256 for n < 2^24 elements, 64 for
n >= 2^24. Evidence (`launch_evidence/validation.json`, `block_search.json`):
256 has the lowest device time on the small/medium search and held-out sizes
(checked end-to-end differences there are within 2%, host-bound); 64 is 1-3%
faster at 2^24-2^25 in every measurement taken (stage-8 diagnostic, device and
end-to-end validation). The per-size timings of this API are those of the
configuration the rule selects; the dispatcher was not timed as a unit.

Safety: the block changes only how elements are partitioned among programs,
not what any element computes (outputs were bitwise identical across blocks in
every run), and `add_relu_wrapper_correctness_block` covers every block the
contract accepts; the fast decision decides `Elementwise2.check` with the block
in use. Not thread-safe (the block is set per call).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_fast as FA  # noqa: E402

LARGE = 2 ** 24


def select_block(n: int) -> int:
    return 64 if n >= LARGE else 256


class CheckedFusedAddReluSelected(FA.CheckedFusedAddReluFast):
    def __init__(self, num_warps: int | None = None):
        super().__init__(block=256, num_warps=num_warps)

    def __call__(self, x, y):
        self.block = select_block(x.numel())
        return super().__call__(x, y)
