/// CPU benchmark: V3 vs V4-VRAM across core counts
mod common;
mod v3;
mod v4_vram;

use common::ScratchPad;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const INPUT_SIZE: usize = 112;

fn main() {
    let hashes_per_core: usize = std::env::args()
        .nth(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let max_cores: usize = std::env::args()
        .nth(2).and_then(|s| s.parse().ok()).unwrap_or(num_cpus::get_physical());

    println!("================================================================");
    println!("XelisHash V3 vs V4-VRAM — CPU Benchmark");
    println!("================================================================");
    println!("VARRAY_SIZE: {} ({:.1} MB)", v4_vram::VARRAY_SIZE,
             v4_vram::VARRAY_SIZE as f64 * 8.0 / 1024.0 / 1024.0);
    println!("VARRAY_READS: {}", option_env!("VARRAY_READS").unwrap_or("2"));
    println!("Hashes per core: {}, Max cores: {}", hashes_per_core, max_cores);
    println!();

    // Correctness
    {
        let mut sp = ScratchPad::default();
        let mut va = Vec::new();
        let h1 = v4_vram::xelis_hash_v4_vram(&[0u8; INPUT_SIZE], &mut sp, &mut va);
        let h2 = v4_vram::xelis_hash_v4_vram(&[0u8; INPUT_SIZE], &mut sp, &mut va);
        assert_eq!(h1, h2, "V4-VRAM not deterministic");
        assert_ne!(h1, [0u8; 32], "V4-VRAM zero output");
        let h3 = v3::xelis_hash_v3(&[0u8; INPUT_SIZE], &mut sp);
        assert_ne!(h1, h3, "V4-VRAM == V3");
        println!("[PASS] V4-VRAM deterministic, differs from V3");
        println!();
    }

    let core_counts: Vec<usize> = [1, 2, 4, 8, 16, 32, max_cores]
        .iter().copied().filter(|&c| c <= max_cores)
        .collect::<std::collections::BTreeSet<usize>>().into_iter().collect();

    println!("{:>6} | {:>10} {:>10} {:>6} | {:>8} {:>8}",
        "Cores", "V3 H/s", "V4V H/s", "V4/V3", "GPU/V3", "GPU/V4V");
    println!("{}", "-".repeat(70));

    let mut results = Vec::new();
    for &cores in &core_counts {
        let total = hashes_per_core * cores;
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let inputs: Vec<[u8; INPUT_SIZE]> = (0..total).map(|_| {
            let mut input = [0u8; INPUT_SIZE]; rng.fill_bytes(&mut input); input
        }).collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(cores).build().unwrap();

        // V3
        let start = Instant::now();
        pool.install(|| inputs.par_iter().for_each(|input| {
            let mut sp = ScratchPad::default();
            std::hint::black_box(v3::xelis_hash_v3(input, &mut sp));
        }));
        let v3_hs = total as f64 / start.elapsed().as_secs_f64();

        // V4-VRAM
        let start = Instant::now();
        pool.install(|| inputs.par_iter().for_each(|input| {
            let mut sp = ScratchPad::default();
            let mut va = Vec::new();
            std::hint::black_box(v4_vram::xelis_hash_v4_vram(input, &mut sp, &mut va));
        }));
        let v4_hs = total as f64 / start.elapsed().as_secs_f64();

        // GPU reference (from VRAM bench, 2MB-2read)
        let gpu_v3 = 23000.0;  // reduced batch due to VRAM
        let gpu_v4 = 14000.0;

        println!("{:>6} | {:>10.1} {:>10.1} {:>5.2}x | {:>7.1}x {:>7.1}x",
            cores, v3_hs, v4_hs, v4_hs / v3_hs,
            gpu_v3 / v3_hs, gpu_v4 / v4_hs);
        results.push((cores, v3_hs, v4_hs));
    }

    println!();
    println!("================================================================");
    println!("GPU (5090, 2MB-2read) vs CPU Summary");
    println!("================================================================");
    for &(cores, v3_hs, v4_hs) in &results {
        println!("{:>2}c: V3 GPU/CPU={:.1}x  V4-VRAM GPU/CPU={:.1}x  improv={:.1}x",
            cores, 23000.0 / v3_hs, 14000.0 / v4_hs, (23000.0/v3_hs) / (14000.0/v4_hs));
    }
}
