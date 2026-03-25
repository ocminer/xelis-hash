# XelisHash V4 — Fair PoW Hash Proposal

## Goal

Equalize mining hashrate across CPUs, GDDR GPUs (RTX 5090), and HBM2 GPUs (CMP 170HX) so that no single platform dominates. Inspired by RandomX/Monero's CPU-friendly approach, adapted for XelisHash's memory-hard foundation.

## Problem with V3

XelisHash V3 heavily favors HBM2 GPUs. The CMP 170HX achieves **258 H/s per SM** with its warp-local memory layout — 1.6x more than RTX 5090 (163 H/s/SM) and ~45x more than a single CPU core. This discourages CPU miners from participating.

## V4 Design — Three Mechanisms

V4 adds three components **on top of the full V3 algorithm** (all V3 memory access patterns preserved):

### 1. Shared-Memory S-box (Occupancy Limiter)

A 2KB per-thread evolving substitution table allocated in GPU shared memory. This limits GPU occupancy to **2 warps/SM** (sm_120) or **1 warp/SM** (sm_80), killing warp-level latency hiding.

```
sbox[256] initialized from scratchpad
Per iteration:
    chain = a ^ sbox[(a >> 56) & 0xFF]    // 1 read
    sbox[(chain >> 48) & 0xFF] ^= chain   // 1 evolving write
```

- **CPU impact**: S-box fits in L1 cache → ~5% overhead
- **GPU impact**: Shared memory pressure limits warps → **2.2x slowdown** on RTX 5090
- **Cannot be fully bypassed**: Moving S-box to global memory recovers only 78% (permanent 22% penalty from traffic overhead). On HBM2 GPUs, bypass is **self-punishing** (slower than shared memory).

### 2. TapeMix (Branch-Prediction Exploit)

A per-hash "instruction tape" of 32 random opcodes derived from the scratchpad. Applied every 32 iterations during stage_3. Each opcode selects from 16 operations of varying cost (XOR to modular_power).

```
tape[32] derived from scratchpad (unique per hash)
Every 32nd iteration:
    for t in 0..32:
        tape_state = tape_exec(tape[t], tape_state, result, operand)
    result ^= tape_state
```

**CPU advantage**: After the first pass through the tape, the CPU branch predictor learns the fixed 32-instruction sequence → near-zero overhead on subsequent iterations (~3 cycles/instruction predicted).

**GPU disadvantage**: 32 threads in a warp each have **different tapes** → warp diverges on every instruction step. With only 2 warps (from S-box), divergence can't be hidden. Adds **31-61% overhead** on top of S-box alone.

Branch prediction cannot be bitsliced, vectorized, or worked around. This is a fundamental CPU hardware advantage.

16 opcodes with varying cost:

| Opcode | Operation | CPU cycles | GPU cycles (divergent) |
|--------|-----------|-----------|----------------------|
| 0-2 | XOR, ADD, ROT | 1-2 | 1-2 (but serialized) |
| 3-5 | MUL, ADD+MUL | 3-6 | 5-10 |
| 6-7 | MurmurHash3, MUL+ROT | 10-15 | 15-20 |
| 8-11 | BLEND, SHIFT, MULHI | 2-5 | 3-8 |
| 12 | ISQRT | ~15 | ~30 (heavy) |
| 13 | AES round | ~1 (AES-NI) | ~20 (software) |
| 14 | Modular power | ~50 | ~100 (heaviest) |
| 15 | 128-bit division | ~20 | ~30 |

### 3. SUPRNOVA Dataset (Multi-Core Scaling Limiter)

A shared 256MB read-only dataset generated deterministically from the seed `0x5355505249564F41` ("SUPRNOVA"). Accessed using a **page-hop pattern**: random 4KB page every 64 iterations, sequential within page.

```
Dataset: 256 MB, 33,554,432 entries
Seed: 0x5355505249564F41 (SUPRNOVA)
Every 64th iteration: hop to random 4KB page
Within page: sequential reads (CPU prefetcher-friendly)
1-2 reads per inner iteration
```

- **CPU**: Hardware prefetcher handles sequential-within-page reads at ~5ns each. Page hops every 64 iters cost one cache miss. Single-core overhead: ~10-20%.
- **GPU**: 32 threads read 32 different random pages → scattered VRAM access. Hidden by existing memory stalls at low occupancy → minimal additional GPU impact.
- **Multi-core scaling**: At 32 cores, 256MB dataset partially overflows L3 (128MB on TR 3970X) → ~50% L3 miss rate on page hops → DRAM latency caps scaling efficiency to ~78%.

Dataset generation (deterministic, same for all miners):
```c
for i in 0..DATASET_SIZE:
    s = SUPRNOVA_SEED ^ (i * 0x9E3779B97F4A7C15)
    s = splitmix64(s)
    if (i & 0xFFF) == 0x53: s ^= SUPRNOVA_SEED  // marker
    dataset[i] = s
```

## Benchmarked Results

All numbers measured on real hardware, not estimated.

### Hardware
- **RTX 5090**: 170 SMs, 32 GB GDDR7, sm_120 — $2000
- **CMP 170HX**: 70 SMs, 8 GB HBM2e, sm_80 — $800
- **TR 3970X**: 32 cores, 128 MB L3, DDR4 quad-channel — $1400

### Configuration
- S-box: 256 entries, 2KB/thread shared memory, MCL=0
- TapeMix: TAPE_LEN=32, TAPE_FREQ=32
- Dataset: 256 MB, DATASET_READS=2, page-hop pattern
- Scratchpad: 544 KB (unchanged from V3)

### Hashrate Comparison

| Platform | V3 kH/s | V4 kH/s | V4/V3 |
|----------|---------|---------|-------|
| RTX 5090 | 27.0 | **4.60** | 5.9x slower |
| CMP 170HX | 18.1 (warp-local) | **2.91** | 6.2x slower |
| CPU 1 core | 0.56 | **0.40** | 1.4x slower |
| CPU 8 cores | 3.97 | **2.76** | 1.4x slower |
| CPU 16 cores | 7.63 | **4.61** | 1.7x slower |
| CPU 32 cores | 13.4 | **8.47** | 1.6x slower |

### Fairness Metrics

| Metric | V3 | V4 | Target |
|--------|----|----|--------|
| GPU/CPU 8-core | 7.0x | **1.67x** | 2-4x |
| GPU/CPU 16-core | 3.5x | **1.00x** | ~1x |
| 5090/CMP per-SM | 0.61x (CMP dominant) | **1.54x** (5090 slight lead) | ~1-2x |

### Price Efficiency

| Platform | Price | V4 kH/s | $/kH/s |
|----------|-------|---------|--------|
| RTX 5090 | $2000 | 4.60 | $435 |
| CMP 170HX | $800 | 2.91 | $275 |
| Ryzen 7 8-core | $300 | 2.76 | $109 |
| Ryzen 9 16-core | $500 | 4.61 | $108 |
| TR 3970X 32-core | $1400 | 8.47 | $165 |

CPUs are the best value per kH/s, making CPU mining profitable and attractive. GPUs remain competitive in absolute hashrate. HBM2 GPUs (CMP 170HX) are no longer dominant.

## Parameters Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| MEMORY_SIZE | 531 × 128 = 67,968 u64s (544 KB) | V3 scratchpad (unchanged) |
| SCRATCHPAD_ITERS | 2 | V3 outer iterations (unchanged) |
| SBOX_ENTRIES | 256 (2 KB/thread) | Occupancy limiter |
| MUL_CHAIN_LEN | 0 | No multiply chain (proven counterproductive) |
| TAPE_LEN | 32 | Instructions per tape |
| TAPE_FREQ | 32 | Apply tape every N iterations |
| DATASET_SIZE | 256 MB (33,554,432 u64s) | Shared read-only dataset |
| DATASET_READS | 2 | Reads per iteration |
| DATASET_SEED | 0x5355505249564F41 | "SUPRNOVA" |
| PAGE_SIZE | 512 u64s (4 KB) | Dataset page-hop granularity |
| PAGE_HOP_FREQ | 64 | New random page every 64 iterations |

## What Was Tried and Rejected

| Approach | Problem |
|----------|---------|
| Multiply chain (MCL=1-32) | Hurts CPU more than GPU (warps absorb compute) |
| S-box cascade (dependent lookups) | Bypassable — L2 cache absorbs S-box on GDDR7 |
| Large V-array (2-4 MB/thread) | Random reads hurt CPU MORE than GPU proportionally |
| BigPad (3-8 MB scratchpad) | Kills CMP 170HX (VRAM 8GB vs 32GB creates 3.3x per-SM gap) |
| AES mixing per iteration | Bypassable via warp-level bitsliced AES |
| Random dataset reads | Too expensive on CPU (DRAM latency dominates tiny CPU iterations) |
| Salsa20/8 compute barrier | GPU warps absorb it for free (1.6% GPU overhead, 168% CPU) |

## Files

- CUDA benchmarks:
  - `xelis_v4_combined_bench.cu` — S-box + TapeMix combined
  - `xelis_v4_dataset_bench.cu` — Full V4 with dataset
  - `xelis_v4_sbox_bench.cu` — S-box parameter sweep
  - `xelis_v4_tape_bench.cu` — TapeMix standalone
  - `xelis_v4_bigpad_bench.cu` — BigPad approach
  - `xelis_v4_vram_bench.cu` — V-array approach
  - `xelis_v4_hybrid_bench.cu` — BigPad + AES hybrid
- CPU benchmarks:
  - `src/bench_full_cpu.rs` — Full V4 with real V3 work + dataset
  - `src/bench_mt.rs` — Multi-threaded S-box benchmark
  - `src/bench_hybrid_cpu.rs` — BigPad + AES CPU
- Sweep results: `sweep_*_results.txt`
- This document: `XELISHASH_V4_PROPOSAL.md`

## Implementation Notes

1. **Dataset generation**: Deterministic from seed, generated once at startup (~1-2 seconds). Same dataset for all miners on the network. Can be regenerated from seed at any time.
2. **S-box**: Allocated in GPU shared memory per-thread. Initialized from scratchpad, evolves every iteration. Must be included in hash computation for correctness.
3. **TapeMix**: Tape derived from scratchpad after stage_1 fill. Same tape for all iterations within a hash. Different tape per hash input.
4. **Verification**: All three components (S-box state, tape state, dataset reads) feed into the final hash output via XOR into `result` and `idx_seed`. Skipping any component produces a different (wrong) hash.

## Variant B: Reduced Scratchpad (448 KB / 480 KB)

To prevent Zen 4+ CPUs (1 MB L2 per core) from having an unfair advantage over Zen 3 (512 KB L2), the scratchpad can be reduced so it fits within 512 KB L2 on **all** modern CPUs:

| Variant | MEMORY_SIZE | Scratchpad | Inner Iters | L2 fit |
|---------|-------------|------------|-------------|--------|
| **A (original)** | 531 × 128 = 67,968 | **544 KB** | 67,968 | Zen 4+ only (1 MB L2) |
| **B-480** | 480 × 128 = 61,440 | **480 KB** | 61,440 | Zen 3+ (512 KB L2) |
| **B-448** | 448 × 128 = 57,344 | **448 KB** | 57,344 | Zen 3+ with headroom |

At 544 KB (Variant A), only Zen 4+ keeps the scratchpad in L2 (~4ns access). Zen 3 overflows to L3 (~15ns) — a ~3.5x latency penalty per random access, giving Zen 4+ a significant unfair edge.

At 448-480 KB (Variant B), the scratchpad fits in Zen 3's 512 KB L2 with room for the S-box (2 KB) and other working data. This equalizes CPU generations so the algorithm favors core count and clock speed, not cache architecture.

**Trade-off**: ~10-15% fewer inner iterations → slightly less memory-hardness. S-box, TapeMix, and dataset parameters remain unchanged. GPU impact is proportional (fewer iterations = proportionally faster on all platforms).

**Recommendation**: Variant B-448 (448 KB) provides the most headroom for S-box + locals while fitting all Zen 3+ and Intel Alder Lake+ L2 caches.

## Next Steps

1. Implement V4 in the XelisHash reference library (Rust)
2. Add V4 CPU verification tests (deterministic test vectors)
3. Implement V4 CUDA kernel in suprminer
4. Pool-test with Suprnova infrastructure
5. Propose to Xelis developers for network upgrade
