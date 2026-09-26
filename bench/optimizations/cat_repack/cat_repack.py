"""Stage-11 candidate B for PyTorch #189940 (dynamic-width QKV repack by
concatenation), pinned for the proof in `CatRepack.lean`. The kernel text is
identical to `scripts/launch_cat.py`'s `repack_fastdiv` (unit test); `magic_for`
computes the host constants (`FastDiv.magic`, `FastDiv.shiftFor` in Lean).
"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["total", "magic", "shift"])
def repack_fastdiv(q1, k1, va1, vb1, q2, k2, va2, vb2, out, wq, wk, wv, total, magic, shift,
                   BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)          # int32: total < 2^31 (checked)
    xm = x < total
    W = 2 * (wq + wk + wv)
    xu = x.to(tl.uint32)
    row = ((tl.umulhi(xu, magic.to(tl.uint32)) + xu) >> shift.to(tl.uint32)).to(tl.int32)
    c = x - row * W
    G = wq + wk + wv
    g2 = c >= G
    cc = tl.where(g2, c - G, c)
    mq = xm & (cc < wq)
    mk = xm & (cc >= wq) & (cc < wq + wk)
    mv = xm & (cc >= wq + wk)
    oq = row * wq + cc
    ok = row * wk + (cc - wq)
    ov = row * wv + (cc - wq - wk)
    q_a = tl.load(q1 + oq, mask=mq & ~g2, other=0.0)
    q_b = tl.load(q2 + oq, mask=mq & g2, other=0.0)
    k_a = tl.load(k1 + ok, mask=mk & ~g2, other=0.0)
    k_b = tl.load(k2 + ok, mask=mk & g2, other=0.0)
    v_a = (tl.load(va1 + ov, mask=mv & ~g2, other=0.0).to(tl.float32)
           + tl.load(vb1 + ov, mask=mv & ~g2, other=0.0).to(tl.float32)).to(out.dtype.element_ty)
    v_b = (tl.load(va2 + ov, mask=mv & g2, other=0.0).to(tl.float32)
           + tl.load(vb2 + ov, mask=mv & g2, other=0.0).to(tl.float32)).to(out.dtype.element_ty)
    val = tl.where(mq, tl.where(g2, q_b, q_a), tl.where(mk, tl.where(g2, k_b, k_a), tl.where(g2, v_b, v_a)))
    tl.store(out + x, val, mask=xm)


def magic_for(d: int) -> tuple[int, int]:
    """ATen IntDivider constants for 32-bit n < 2^31: (m1, shift) with
    n // d == (umulhi(n, m1) + n) >> shift."""
    shift = 0
    while (1 << shift) < d:
        shift += 1
    m1 = ((1 << 32) * ((1 << shift) - d)) // d + 1
    return m1, shift
