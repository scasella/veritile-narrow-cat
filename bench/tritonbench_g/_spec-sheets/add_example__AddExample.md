# Spec sheet — `bench/tritonbench_g/add_example/AddExample.lean`

**Python source:** `bench/tritonbench_g/add_example/add_example.py`

## Public theorem: `add_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `add_kernel` implements pointwise addition on its masked
IO signature — for every disjoint flat placement of the three buffers, every
program id whose active lanes are in bounds, and every launch state whose
input windows hold `xs`/`ys` at the active lanes, the translated pointer
kernel terminates, every active output lane holds `xs i + ys i`, and every
other memory cell is unchanged. Proof: `MaskedKernelIO₂.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
```
</details>

**Statement:**
```lean
specification add_kernel_correctness
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    addIO in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => xs i + ys i
```

**Closed-form spec defs (transitive):** `addIO`, `add_kernel`

<details><summary><code>addIO</code></summary>

```
/-- `add_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `pid` reads its input tiles / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`offsets = pid * BLOCK_SIZE + arange;
mask = offsets < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
```
```lean
def addIO (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  in1 := in_ptr0
  in2 := in_ptr1
  out := out_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `add_example.py`'s `add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` annotation → Lean `Nat` parameter
  (the `tl.constexpr` is implicit in Lean params).

Everything else is verbatim from the upstream kernel. -/
```
```lean
def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>

## Public theorem: `add_kernel_launch_correctness`

<details><summary>docstring</summary>

```
/-- **Whole-launch headline.** For a host configuration accepted by the proved
launch checker, launching `add_kernel` over the host's grid `c.grid` from any
state whose input cells `i < n` hold typed real values `xs i`, `ys i`:
every program terminates and is trace-safe for bounds covering the checked
capacities; the programs' write sets are pairwise disjoint and compose into
one final memory (`GridLaunchedOrdinary`) in which `out_ptr[i] = xs i + ys i`
for every `i < n`, while every other cell is unchanged; and, as arithmetic
facts about `c`, every lane offset and mask evaluated in two's-complement
`i32` equals its ℕ counterpart used by the model (the model itself computes
over ℕ; that Triton evaluates these expressions in `i32` is the trusted
assumption TA-i32). Obligations used: P1–P3 (first conjunct), P4 (second),
P5–P7 (third); P8–P10 justify the translation assumptions TA-region and
TA-compose (`CONTRACT.md` §2). -/
```
</details>

**Statement:**
```lean
specification add_kernel_launch_correctness
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (bx by_ : BufMeta) (hin : c.inputs = [bx, by_])
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
        { dims := c.grid } s
        (fun i : Nat => if i < c.n then some (out_ptr, i) else none)
        (fun i => xs i + ys i) ∧
      (∀ bounds : RegionBounds,
        bx.capacity ≤ bounds in_ptr0 → by_.capacity ≤ bounds in_ptr1 →
        c.output.capacity ≤ bounds out_ptr →
        ∀ idx : GridIndex { dims := c.grid },
          Kernel.TraceSafe bounds
            ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
            (s.withGridIndex idx)) ∧
      ∀ pid j, pid < c.gridX → j < c.block →
        (BitVec.ofNat 32 pid * BitVec.ofNat 32 c.block + BitVec.ofNat 32 j).toInt
            = ((pid * c.block + j : Nat) : Int) ∧
        BitVec.slt (BitVec.ofNat 32 pid * BitVec.ofNat 32 c.block + BitVec.ofNat 32 j)
            (BitVec.ofNat 32 c.n) = decide (pid * c.block + j < c.n)
```

**Assumptions / layout contracts:**
- `hc : Blocked1DLaunch.check c = Bool.true`
- `hin : c.inputs = [bx, by_]`
- `xs ys : Nat → ℝ`
- `hx : ∀ i, i < c.n → s.mem in_ptr0 i = MemCell.real (xs i)`
- `hy : ∀ i, i < c.n → s.mem in_ptr1 i = MemCell.real (ys i)`

**Closed-form spec defs (transitive):** `add_kernel`

<details><summary><code>add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `add_example.py`'s `add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` annotation → Lean `Nat` parameter
  (the `tl.constexpr` is implicit in Lean params).

Everything else is verbatim from the upstream kernel. -/
```
```lean
def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>

## Public theorem: `add_wrapper_correctness`

<details><summary>docstring</summary>

```
/-- **Whole-wrapper headline.** For tensors accepted by the wrapper checker
and region names that follow the allocations, from any state whose input
tensors hold typed real values:

1. every element `i < out.numel` of the returned tensor is written with
   `xs i + ys i`, and every other cell is unchanged (framed whole-grid launch);
2. for every in-shape multi-index, the three tensors address the same
   row-major element, below `out.numel` (the logical view);
3. every program is trace-safe for bounds equal to the tensors' own element
   counts — no access leaves an input's logical extent;
4. running the programs one after another in **any** complete order gives the
   same result (so the merge semantics does not rely on a program never
   reading another's writes);
5. in the flat memory placed at the tensors' element addresses
   (`base / elemBytes`), the checked conditions discharge the bridge's
   disjointness and closure hypotheses and the flattened launch writes
   `xs i + ys i` at element address `out.base / out.elemBytes + i` for every
   `i < out.numel`, leaving every other flat cell unchanged. -/
```
</details>

**Statement:**
```lean
specification add_wrapper_correctness
    (x y out : TensorMeta) (hc : Elementwise2.check 4 x y out = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (h0 : out_ptr ≠ in_ptr0) (h1 : out_ptr ≠ in_ptr1)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < x.numel → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < y.numel → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((add_kernel in_ptr0 in_ptr1 out_ptr x.numel 4).toAlgKernel)
        { dims := (Elementwise2.launch 4 x y out).grid } s
        (fun i : Nat => if i < out.numel then some (out_ptr, i) else none)
        (fun i => xs i + ys i) ∧
      (∀ idx, TensorMeta.InShape out.shape idx →
        out.offsetOf idx = x.offsetOf idx ∧ y.offsetOf idx = x.offsetOf idx ∧
          x.offsetOf idx < out.numel) ∧
      (∀ bounds : RegionBounds,
        x.numel ≤ bounds in_ptr0 → y.numel ≤ bounds in_ptr1 → out.numel ≤ bounds out_ptr →
        ∀ idx : GridIndex { dims := (Elementwise2.launch 4 x y out).grid },
          Kernel.TraceSafe bounds
            ((add_kernel in_ptr0 in_ptr1 out_ptr x.numel 4).toAlgKernel)
            (s.withGridIndex idx)) ∧
      (∀ L : List (GridIndex { dims := (Elementwise2.launch 4 x y out).grid }),
        L.Nodup → (∀ idx, idx ∈ L) →
        ∃ m, Kernel.runSerial ((add_kernel in_ptr0 in_ptr1 out_ptr x.numel 4).toAlgKernel)
              s L s.mem = some m ∧
          (∀ i, i < out.numel → Kernel.memReal m out_ptr i = xs i + ys i) ∧
          (∀ r o, ¬ (r = out_ptr ∧ o < out.numel) → m r o = s.mem r o)) ∧
      (∀ flat : RegionName, (in_ptr0 = in_ptr1 ↔ x.base = y.base) →
        s.undef = (fun _ _ => 0) →
        (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).Disjoint ∧
        (∀ r, r ∉ (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).regions →
          (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).extent r = 0) ∧
        Kernel.LaunchCorrectFramed
          ((Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).flattenKernel
            ((add_kernel in_ptr0 in_ptr1 out_ptr x.numel 4).toAlgKernel))
          { dims := (Elementwise2.launch 4 x y out).grid }
          ((Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).flattenState s)
          (fun i : Nat => if i < out.numel then
            some (flat, out.base / out.elemBytes + i) else none)
          (fun i => xs i + ys i))
```

**Assumptions / layout contracts:**
- `hc : Elementwise2.check 4 x y out = Bool.true`
- `h0 : out_ptr ≠ in_ptr0`
- `h1 : out_ptr ≠ in_ptr1`
- `xs ys : Nat → ℝ`
- `hx : ∀ i, i < x.numel → s.mem in_ptr0 i = MemCell.real (xs i)`
- `hy : ∀ i, i < y.numel → s.mem in_ptr1 i = MemCell.real (ys i)`

**Closed-form spec defs (transitive):** `add_kernel`

<details><summary><code>add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `add_example.py`'s `add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` annotation → Lean `Nat` parameter
  (the `tl.constexpr` is implicit in Lean params).

Everything else is verbatim from the upstream kernel. -/
```
```lean
def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>
