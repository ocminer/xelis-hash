/// Multi-threaded CPU benchmark: V3 vs V4 across core counts and MUL_CHAIN_LEN values.
/// Usage: bench_mt [hashes_per_core] [max_cores]
mod common;
mod v3;
mod v4;

use common::ScratchPad;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const INPUT_SIZE: usize = 112;

fn bench_v3_mt(inputs: &[[u8; INPUT_SIZE]], num_cores: usize) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_cores)
        .build()
        .unwrap();
    let start = Instant::now();
    pool.install(|| {
        inputs.par_iter().for_each(|input| {
            let mut sp = ScratchPad::default();
            std::hint::black_box(v3::xelis_hash_v3(input, &mut sp));
        });
    });
    let elapsed = start.elapsed();
    inputs.len() as f64 / elapsed.as_secs_f64()
}

fn bench_v4_mt(inputs: &[[u8; INPUT_SIZE]], num_cores: usize) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_cores)
        .build()
        .unwrap();
    let start = Instant::now();
    pool.install(|| {
        inputs.par_iter().for_each(|input| {
            let mut sp = ScratchPad::default();
            std::hint::black_box(v4::xelis_hash_v4(input, &mut sp));
        });
    });
    let elapsed = start.elapsed();
    inputs.len() as f64 / elapsed.as_secs_f64()
}

fn main() {
    let hashes_per_core: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let max_cores: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_cpus::get_physical());

    let mcl = option_env!("MUL_CHAIN_LEN").unwrap_or("16");

    println!("================================================================");
    println!("XelisHash V3 vs V4 — Multi-Core CPU Benchmark");
    println!("================================================================");
    println!("MUL_CHAIN_LEN: {}", mcl);
    println!("Physical cores: {}", num_cpus::get_physical());
    println!("Hashes per core: {}", hashes_per_core);
    println!("Max cores tested: {}", max_cores);
    println!();

    // Verify correctness first
    {
        let mut sp = ScratchPad::default();
        let h3 = v3::xelis_hash_v3(&[0u8; INPUT_SIZE], &mut sp);
        let expected = [
            105, 172, 103, 40, 94, 253, 92, 162, 42, 252, 5, 196, 236, 238, 91, 218, 22, 157,
            228, 233, 239, 8, 250, 57, 212, 166, 121, 132, 148, 205, 103, 163,
        ];
        assert_eq!(h3, expected, "V3 test vector MISMATCH");
        let h4 = v4::xelis_hash_v4(&[0u8; INPUT_SIZE], &mut sp);
        assert_ne!(h3, h4, "V3 == V4 (should differ)");
        println!("[PASS] V3 test vector + V4 determinism verified");
        println!();
    }

    // Core counts to test
    let core_counts: Vec<usize> = [1, 2, 4, 8, 16, 32, max_cores]
        .iter()
        .copied()
        .filter(|&c| c <= max_cores)
        .collect::<std::collections::BTreeSet<usize>>()
        .into_iter()
        .collect();

    // GPU reference numbers
    let gpu_5090_v3 = 27000.0_f64;  // H/s
    let gpu_5090_v4_mcl0 = 12280.0;
    let gpu_5090_v4_mcl16 = 11670.0;

    println!("{:>6} | {:>10} {:>10} {:>6} | {:>10} {:>10} {:>6} | {:>8} {:>8}",
        "Cores", "V3 H/s", "V4 H/s", "V4/V3", "V3 scal", "V4 scal", "eff%",
        "GPU/V3", "GPU/V4");
    println!("{}", "-".repeat(100));

    let mut results: Vec<(usize, f64, f64)> = Vec::new();

    for &cores in &core_counts {
        let total_hashes = hashes_per_core * cores;
        // Generate deterministic random inputs
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut inputs: Vec<[u8; INPUT_SIZE]> = Vec::with_capacity(total_hashes);
        for _ in 0..total_hashes {
            let mut input = [0u8; INPUT_SIZE];
            rng.fill_bytes(&mut input);
            inputs.push(input);
        }

        // Warmup
        if cores == core_counts[0] {
            let _ = bench_v3_mt(&inputs[..cores.min(4)], cores.min(4));
            let _ = bench_v4_mt(&inputs[..cores.min(4)], cores.min(4));
        }

        let v3_hs = bench_v3_mt(&inputs, cores);
        let v4_hs = bench_v4_mt(&inputs, cores);

        let v3_scaling = if !results.is_empty() { v3_hs / results[0].1 } else { 1.0 };
        let v4_scaling = if !results.is_empty() { v4_hs / results[0].2 } else { 1.0 };
        let efficiency = if cores > 1 { v3_scaling / cores as f64 * 100.0 } else { 100.0 };

        println!("{:>6} | {:>10.1} {:>10.1} {:>5.2}x | {:>9.1}x {:>9.1}x {:>5.1}% | {:>7.1}x {:>7.1}x",
            cores, v3_hs, v4_hs, v4_hs / v3_hs,
            v3_scaling, v4_scaling, efficiency,
            gpu_5090_v3 / v3_hs, gpu_5090_v4_mcl0 / v4_hs);

        results.push((cores, v3_hs, v4_hs));
    }

    println!();
    println!("================================================================");
    println!("GPU vs CPU Summary (RTX 5090 vs TR 3970X)");
    println!("================================================================");
    for &(cores, v3_hs, v4_hs) in &results {
        let gpu_v3_ratio = gpu_5090_v3 / v3_hs;
        let gpu_v4_ratio = gpu_5090_v4_mcl0 / v4_hs;
        println!("{:>2} cores: GPU/CPU V3={:.1}x  V4(MCL=0)={:.1}x  improvement={:.1}x",
            cores, gpu_v3_ratio, gpu_v4_ratio, gpu_v3_ratio / gpu_v4_ratio);
    }
    println!();

    // Final analysis
    if let Some(&(_, _, v4_hs_max)) = results.last() {
        println!("RTX 5090 V4 (MCL=0): {:.0} H/s", gpu_5090_v4_mcl0);
        println!("TR 3970X V4 (MCL=0, {} cores): {:.0} H/s", max_cores, v4_hs_max);
        println!("GPU/CPU ratio: {:.2}x", gpu_5090_v4_mcl0 / v4_hs_max);
    }
}
