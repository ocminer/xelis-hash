# XelisHashV4 Design Document — GPU Warp Neutralization

## Status: ACTIVE — benchmarking on RTX 5090 + CMP 170HX

## Goal

Equalize hashrate across:
1. HBM2 vs GDDR7 (same generation GPU) — eliminate memory-type advantage
2. CPU vs GPU (same generation) — narrow gap to ~2-4x

## Why V3 Favors GPUs (Root Cause)

V3's stage 3: 67,968 iterations, each with ~6 random 8-byte memory ops and ~20 ALU ops.
GPUs run 16-40+ warps per SM. When one warp stalls on memory (~400 cycles), another
executes. The ~20 ALU ops fit into "free" slots — compute is completely hidden.

**Proof**: Adding Salsa20/8 (400 ALU ops) per iteration was invisible on RTX 5090
(1.6% slowdown) because warps absorbed it. But on CPU (1 thread, no warp scheduling)
it caused 168% slowdown. Adding compute alone cannot equalize.

## The Solution: Starve the SM of Warps

GPUs hide memory latency by switching warps. Kill the warps, kill the advantage.

**Three mechanisms in concert:**

### Mechanism 1: Shared Memory S-box (2 KB/thread) — Occupancy Killer

Each thread owns a 2048-byte evolving substitution table in shared memory.

| GPU | Shared Mem/SM | Max Threads | Max Warps |
|-----|---------------|-------------|-----------|
| sm_120 (RTX 5090) | 128 KB | 64 | **2** |
| sm_80 (CMP 170HX) | 96 KB (max config) | 48 → 32 | **1** |

With only 1-2 warps per SM, the GPU cannot hide memory latency.
The S-box **evolves** every iteration (1 entry XOR'd with result),
preventing warp-sharing or precomputation tricks.

**CPU impact**: 2 KB fits in L1 cache. 4-cycle access. Negligible.

### Mechanism 2: Serial Multiply Chain (16-32 chained mul64) — Latency Floor

After the branch operation, a chain of dependent 64-bit multiplies:
```
chain = a ^ sbox_val
chain *= (si | 1)
chain *= (rotr(chain, 13) ^ sj)
chain *= (result ^ (chain >> 32))
... (16 total dependent multiplies)
```

Each mul64: **5-6 cycles on GPU** (MUL.WIDE.U32, not pipelined for dependent ops).
Each mul64: **3 cycles on CPU** (IMUL r64, fully pipelined).

16 chained mul64:
- GPU: 16 × 5 = **80 serial cycles** per iteration per warp
- CPU: 16 × 3 = **48 serial cycles** per iteration

With 2 warps: 2 × 80 = 160 cycles of compute vs ~800 cycles of memory stall.
Coverage: 20%. The SM sits idle 80% of the time. Warps are useless.

### Mechanism 3: Reduced Memory Ops (2 per iteration, not 6)

V3 does 6 random memory ops per iteration (a, b, c_random, t, old_a, old_b + 2 stores).
V4 does 2 random memory ops (1 read + 1 RMW) plus 1 S-box read + 1 S-box write.

Fewer memory ops means the multiply chain represents a larger fraction of total work.
The S-box ops hit shared memory (20-30 cycle latency), not global DRAM.

### Mechanism 4: Persistent Register State (8 × u64)

8 state words (s0..s7) live in registers across all iterations.
Used as mixing inputs to the multiply chain.
Updated every iteration: `state[idx] = chain`.

Forces ~80 registers/thread. On sm_120, 65536/80 = 819 threads = 25 warps max
from registers alone. But shared memory (mechanism 1) is the tighter constraint
at 2 warps, so registers are not the bottleneck — they just prevent occupancy tricks.

## V4 Stage 3 Inner Loop

```
Constants:
  MEMORY_SIZE   = 67,968 u64s (544 KB, unchanged)
  BUFFER_SIZE   = 33,984 (unchanged)
  SCRATCHPAD_ITERS = 2 (unchanged)
  MUL_CHAIN_LEN = 16 (tunable: 16-32)
  SBOX_ENTRIES  = 256 (2 KB per thread)
  STATE_WORDS   = 8

Initialization (per hash):
  sbox[0..255] = scratchpad[0..255] ^ blake3_derived_key
  state[0..7]  = scratchpad[256..263]

Per outer iteration (×2):
  AES round → result (same as V3)

  Per inner iteration j (×33,984):
    // PHASE 1: One random global read + S-box lookup
    idx = map_index(result)
    a = scratchpad_a[idx]
    sbox_val = sbox[(a >> 56) & 0xFF]   // shared mem on GPU, L1 on CPU

    // PHASE 2: Gather state words
    si = state[(result >> 4) & 7]
    sj = state[(result >> 7) & 7]

    // PHASE 3: Serial multiply chain (16 dependent mul64)
    chain = a ^ sbox_val
    for k in 0..MUL_CHAIN_LEN:
        operand = select mixing input based on k (si, sj, result, sbox_val, a, rotations of chain)
        chain *= (operand | 1)  // |1 prevents degenerate zero-chains

    // PHASE 4: State + S-box update
    state[(result >> 10) & 7] = chain
    sbox[(chain >> 48) & 0xFF] ^= chain

    // PHASE 5: One random global RMW
    widx = map_index(chain ^ result ^ CONST1)
    old = scratchpad_b[widx]
    scratchpad_b[widx] = old ^ chain
    result = rotl64(chain ^ old, r)
    r = (r < MEMORY_SIZE - 1) ? r + 1 : 0
```

## Performance Projections

### RTX 5090 (sm_120, 170 SMs)

With 2 warps/SM, 2 global mem ops/iter at ~400 cycles each:
- Memory stall per iter: ~800 cycles (serial, dependent addresses)
- Compute per iter: 2 warps × 80 cycles = 160 cycles
- Effective cycles/iter: ~800 (memory-dominated, can't hide)
- Total: 33,984 × 800 × 2 = ~54.4M cycles/hash
- Per SM at ~2.1 GHz: ~25.9 ms/hash
- 170 SMs: ~6,560 H/s (6.56 kH/s)
- **Down from 26.76 kH/s** (4.1× reduction)

### CMP 170HX (sm_80, 70 SMs)

With 1 warp/SM (96KB shared / 2KB/thread = 48, but 1 warp = 32):
- Memory stall per iter: ~600 cycles (HBM2 lower latency)
- Compute per iter: 1 warp × 80 = 80 cycles
- Effective cycles/iter: ~600 (zero hiding)
- Total: 33,984 × 600 × 2 = ~40.8M cycles/hash
- Per SM at ~1.4 GHz: ~29.1 ms/hash
- 70 SMs: ~2,405 H/s (2.4 kH/s)

**HBM2 advantage**: 6560/2405 × (70/170) = ~1.12×/SM. Nearly equalized.
The residual difference is pure SM count × clock, not memory type.

### CPU — Threadripper 3970X (32 cores)

- 544 KB scratchpad fits in L3 (128 MB), ~30-80 cycle random access
- 2 KB S-box fits in L1, ~4 cycle access
- 16 mul64 = 48 cycles (3 cycles/mul, x86 IMUL)
- Per iteration: ~50 (L3) + 4 (S-box) + 48 (multiply) + ~20 (overhead) ≈ 122 cycles
- Total: 33,984 × 122 × 2 = ~8.3M cycles/hash
- At 3.7 GHz: ~2.24 ms/hash = ~446 H/s per core
- 32 cores: ~14,272 H/s

**GPU/CPU ratio**: 6560 / 14272 = **0.46×**. CPU actually wins slightly!

### Tuning MUL_CHAIN_LEN

| Chain Length | GPU (5090) | CPU (32-core) | Ratio |
|-------------|------------|---------------|-------|
| 8 | ~8.5 kH/s | ~17 kH/s | 0.50 |
| 16 | ~6.5 kH/s | ~14 kH/s | 0.46 |
| 24 | ~5.5 kH/s | ~12 kH/s | 0.46 |
| 32 | ~4.8 kH/s | ~10 kH/s | 0.48 |

The ratio is stable across chain lengths because both CPU and GPU scale similarly
with multiply count. The dominant factor is the occupancy limit (2 warps) which
is controlled by shared memory, not the chain length.

**Recommended: MUL_CHAIN_LEN = 16** (good balance, not excessive CPU overhead)

## Security Analysis

### Memory-Hardness: Preserved
- Same 544 KB scratchpad with random read and random RMW per iteration
- 67,968 total iterations touch scratchpad (same as V3)
- Scratchpad cannot be skipped (result depends on loaded values)

### Serial Dependency: Strengthened
- V3: result[j+1] = f(scratchpad[random], result[j])
- V4: result[j+1] = f(scratchpad[random], sbox[random], multiply_chain(result[j], state[]))
- The multiply chain adds 16 non-invertible transformations to the chain

### S-box Evolution: New Defense
- S-box state changes every iteration: `sbox[idx] ^= chain`
- After 33,984 iterations, every entry has been modified ~133 times on average
- This creates a per-hash-instance "derived key" that can't be precomputed
- Prevents lookup table sharing between threads/warps

### No Shortcuts
- Skipping the multiply chain produces wrong indices → wrong hash
- Skipping S-box reads produces wrong chain values → wrong hash
- The S-box evolves based on chain output, which depends on S-box input → circular dependency

## Benchmark Results (2026-03-22)

### Attempt 1: Salsa20/8 compute barrier only (FAILED — warps absorbed it)

| | V3 (kH/s) | V4-salsa (kH/s) | Overhead |
|---|---|---|---|
| RTX 5090 | 27.08 | 26.66 | **1.6%** — invisible |
| CMP 170HX | 4.28 | 4.03 | **6.1%** — minimal |
| TR 3970X (CPU, 1 core) | 567 H/s | 211 H/s | **168%** — devastating |

Conclusion: Adding compute alone doesn't work. GPUs hide it via warp scheduling.

### Attempt 2: Shared-mem S-box + full V3 memory pattern + mul chain (WORKING)

| | V3 (kH/s) | V4-sbox (kH/s) | Slowdown | Per-SM (H/s) |
|---|---|---|---|---|
| RTX 5090 (170 SMs) | 27.0 | **11.7** | **2.31x** | 68.6 |
| CMP 170HX (70 SMs) | 4.42* | **4.16** | **1.06x** | 59.4 |

*CMP V3 without warp-local layout (production V3: 18.1 kH/s with warp-local)

**Per-SM equalization: 68.6 / 59.4 = 1.15x** (down from 2.5x in V3).
The remaining 15% difference is clock speed + arch, not memory type.

### CPU Benchmark (Rust, full hash — needs V4 Rust update for new design)

| CPU | V3 (H/s) | V4-salsa (H/s) | V4-sbox (H/s) |
|-----|----------|----------|----------|
| TR 3970X (1 core) | 567 | 211 | TBD (expect ~400-500 — S-box in L1, mul64 is fast) |

## Implementation Files

```
/home/marcel/xelis-hash/V4/
  xelishashv4-design-doc.md     ← this file
  DESIGN.md                     ← original V4 attempt (Salsa20/8, failed)
  xelis_v3v4_bench.cu           ← standalone CUDA benchmark
  Cargo.toml                    ← Rust CPU benchmark project
  src/common.rs                 ← shared scratchpad, stage_1, stage_4
  src/v3.rs                     ← V3 reference
  src/v4.rs                     ← V4 CPU implementation (to be updated)
  src/main.rs                   ← CPU benchmark runner
```

## Iteration Log

### Attempt 1: Salsa20/8 Compute Barrier (FAILED)
- Added ~400 ALU ops (Salsa20/8 + 4 mul64) per iteration
- RTX 5090: 1.6% overhead (invisible — warps absorbed it)
- CMP 170HX: 6.1% overhead (still minimal)
- CPU: 168% overhead (no warp hiding)
- **Conclusion**: Adding compute alone doesn't work. Must reduce warp count.

### Attempt 2a: Shared Memory + Multiply Chain, REDUCED memory ops (WRONG DIRECTION)
- 2 KB/thread shared S-box → max 2 warps/SM
- Reduced to 2 global mem ops per iteration (from 6)
- **Result: V4 was 1.68x FASTER than V3 (45 kH/s vs 27 kH/s)**
- Fewer memory ops made it less memory-intensive. Wrong direction.

### Attempt 2b: Shared Memory + Multiply Chain, FULL V3 memory pattern (SUCCESS)
- Keep ALL V3 memory ops (6 random global accesses per iteration)
- ADD S-box lookup + multiply chain ON TOP
- S-box forces 2 warps/SM max via shared memory pressure
- **RTX 5090: 27.0 → 11.7 kH/s (2.31x slower). Per-SM equalized to 1.15x.**
- Next: update CPU benchmark, tune MUL_CHAIN_LEN
