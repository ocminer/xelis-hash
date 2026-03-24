# XelisHashV4 Design Document

## Goal

Equalize hashrate across memory types (HBM2 vs GDDR7) and narrow the CPU-GPU gap,
so that no single hardware class has a disproportionate advantage.

## Problem Analysis

### Why HBM2 dominates XelisHashV3

XelisHashV3's stage 3 inner loop is **memory-latency bound**:

```
result[j] -> map_index -> random load a -> compute b_index -> random load b
          -> trivial ALU (branch op) -> random RMW -> result[j+1]
```

Per iteration: ~6 random memory accesses, ~20 ALU ops.
The GPU spends ~95% of time waiting for memory.

**Bandwidth per SM** determines throughput:
- CMP 170HX (HBM2): 1493 GB/s / 70 SMs = **21.3 GB/s per SM**
- RTX 5090 (GDDR7): 1792 GB/s / 170 SMs = **10.5 GB/s per SM**

HBM2 has **2x bandwidth per SM**, translating directly to ~2x hashrate per SM.

### Why YescryptR32 does NOT favor HBM2

YescryptR32's inner loops are **compute-bound**:
- Salsa20/8: 256 ALU ops (add32, xor32, rotate32) per call
- PWXform: 6 rounds x 8 multiplies (64-bit) per blockmix iteration
- 64 blockmix iterations per fill/mix step = ~22,784 ops per blockmix

The GPU spends >80% of time doing integer math. Memory bandwidth is irrelevant.

### Key Insight

To equalize memory types: make each inner-loop iteration **compute-heavy enough**
that memory latency becomes a minor fraction of total iteration time.

If `compute_time >> memory_latency`, then:
- Throughput = f(ALU_capacity), not f(memory_bandwidth)
- HBM2 advantage vanishes
- CPUs remain competitive (strong per-core ALU, scratchpad fits in L3)

## V4 Design: Compute Barrier

### What changes from V3

Only **stage 3** is modified. Stages 1 (ChaCha8 fill) and 4 (Blake3 hash) are identical.

In V3's inner loop, after computing the branch result `v`, the algorithm immediately
uses `v` to derive the next `result`. In V4, we inject a **compute barrier** between
`v` and `result`:

```
V3: v = branch_op(...)  ->  seed = v ^ result  ->  result = rotl(seed, r)
V4: v = branch_op(...)  ->  hardened = compute_barrier(v, result, a, b, c)
                         ->  seed = hardened ^ result  ->  result = rotl(seed, r)
```

The barrier is in the **serial dependency chain**: `result[j+1]` depends on
`compute_barrier()` output, which depends on `v`, which depends on memory loads
from iteration `j`. No iteration can begin until the previous barrier completes.

### Compute Barrier: Salsa20/8 + Serial Multiply Chain

```
compute_barrier(v, result, a, b, c) -> u64:

  1. EXPAND: Pack 5 x u64 inputs into 16 x u32 Salsa state
     state[0..1]   = v
     state[2..3]   = result
     state[4..5]   = a
     state[6..7]   = b
     state[8..9]   = c
     state[10..11] = v ^ a          (cross-mixing)
     state[12..13] = result ^ b     (cross-mixing)
     state[14..15] = c ^ !result    (asymmetric mixing)

  2. SALSA20/8: 4 double-rounds (8 rounds total)
     32 quarter-rounds, each: 3 add + 3 rotate + 3 xor = 12 ops
     Total: 384 ALU ops
     + 16 feedforward additions

  3. SERIAL MULTIPLY CHAIN: 4 iterations of
     m = (m_lo * m_hi) + state_word
     Each 64-bit multiply is 1 cycle on CPU, 4-8 cycles on GPU.
     Forces serial dependency that can't be parallelized.
     Total: 4 mul64 + 4 add64

  4. FOLD: XOR remaining 8 state words into result
     Total: 7 XOR operations

  Output: m ^ tail
```

### Why Salsa20/8?

1. **Well-studied**: 15+ years of cryptanalysis, known security margins
2. **Register-friendly**: Operates on 16 x u32 — fits in GPU registers without spills
3. **No memory access**: Pure ALU, doesn't add memory pressure
4. **Fair across architectures**: add/xor/rotate are 1-cycle on both CPU and GPU
5. **Non-invertible with feedforward**: output = f(input) + input prevents inversion

### Why the Serial Multiply Chain?

The Salsa20/8 core has **internal parallelism** — quarter-rounds on independent
columns/rows can execute simultaneously on superscalar CPUs or wide GPU SMs.

The multiply chain is **strictly serial**:
```
m = lo(m) * hi(m) + k1
m = lo(m) * hi(m) + k2    // depends on previous m
m = lo(m) * hi(m) + k3    // depends on previous m
m = lo(m) * hi(m) + k4    // depends on previous m
```

This ensures a minimum latency floor that can't be reduced by wider hardware.
On CPU: 4 cycles (1-cycle multiply latency). On GPU: 16-32 cycles (4-8 cycle MUL.WIDE).

### Operation Count Per Inner Iteration

| Component | Ops | CPU cycles (est.) | GPU cycles (est.) |
|-----------|-----|-------------------|-------------------|
| V3 memory loads (a,b,c,t,old_a,old_b) | 6 loads | 30-60 (L3 hit) | 400-800 (DRAM) |
| V3 branch ALU | ~20 | 5-10 | 20-40 |
| **V4 Salsa20/8** | **384+16** | **~100** | **~200** |
| **V4 multiply chain** | **4 mul + 4 add** | **~8** | **~32** |
| **V4 expand + fold** | **~30** | **~8** | **~15** |
| **V4 total added** | **~434** | **~116** | **~247** |

### Expected Impact

**Before (V3)**:
- GPU iteration time: ~95% memory, ~5% compute
- HBM2 advantage: ~2x (bandwidth/SM ratio)

**After (V4)**:
- GPU iteration time: ~60-70% memory, ~30-40% compute
- HBM2 advantage: ~1.2-1.4x (diminished)

**CPU impact**:
- V3 iteration: ~50 cycles (L3-cached scratchpad + trivial ALU)
- V4 iteration: ~170 cycles (same memory + barrier)
- Slowdown: ~3.4x per hash
- But the GPU slowdown is similar (~2-3x), preserving CPU competitiveness

### Why Not More Compute?

Adding too much compute would:
1. Make the algorithm purely compute-bound (loses memory-hardness value)
2. Favor ASICs with custom ALU pipelines
3. Make CPUs uncompetitively slow (they can't parallelize across hashes like GPUs)

The Salsa20/8 barrier is calibrated to shift ~30-40% of iteration time to compute,
enough to significantly diminish the HBM2 advantage without destroying memory-hardness.

## Security Analysis

### Memory-Hardness Preserved
- Same 544 KB scratchpad with random read-modify-write
- Same 67,968 total inner iterations (33,984 x 2 outer)
- Same index derivation via MurmurHash3 + multiply-high
- No memory access patterns changed

### Serial Dependency Preserved
- compute_barrier output feeds into `seed`, which feeds into `result`
- `result` determines next iteration's memory indices
- No iteration can overlap — the chain is unbreakable

### No New Attack Surface
- Salsa20/8 is a one-way function (with feedforward)
- The multiply chain is non-invertible (truncation via u32 cast)
- All 5 inputs (v, result, a, b, c) are incorporated
- Skipping the barrier produces wrong indices → wrong hash

### Strengthening Effect
V4 is strictly harder than V3:
- Same memory work PLUS additional compute
- The barrier adds mixing that makes `result` depend on more state
- Finding collisions or preimages is at least as hard as V3

## Benchmark Design

The benchmark compares V3 and V4 on random inputs:

1. Pre-generate N random 112-byte inputs (prevents cache reuse between hashes)
2. Verify V3 against known test vector (correctness check)
3. Verify V4 determinism (same input -> same output)
4. Time N full hashes for each version
5. Time stage_3 only (with fresh scratchpad per call) for accurate overhead measurement
6. Report H/s, ratio, and per-stage breakdown

### Running

```bash
cd V4
cargo build --release
./target/release/xelishash-v4-bench          # default 100 hashes
./target/release/xelishash-v4-bench 500      # 500 hashes
```

## File Structure

```
V4/
├── DESIGN.md           (this document)
├── Cargo.toml          (standalone project)
├── src/
│   ├── main.rs         (benchmark runner)
│   ├── common.rs       (scratchpad, stage_1, stage_4, helpers)
│   ├── v3.rs           (V3 stage_3 reference copy)
│   └── v4.rs           (V4 stage_3 with compute barrier)
```
