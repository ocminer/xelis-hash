/// CPU benchmark: simulate larger scratchpad by running V3 stage_3 PAD_MULT times.
/// This is conservative — actual larger-pad V3 would have different access patterns
/// but same cache behavior since pad still fits L3.
mod common;
mod v3;

use common::ScratchPad;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const INPUT_SIZE: usize = 112;

fn main() {
    let pad_mult: usize = std::env::args()
        .nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let hashes: usize = std::env::args()
        .nth(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let max_cores: usize = std::env::args()
        .nth(3).and_then(|s| s.parse().ok()).unwrap_or(num_cpus::get_physical());

    println!("================================================================");
    println!("BigPad CPU Benchmark — PAD_MULT={} (simulated)", pad_mult);
    println!("================================================================");
    println!("Effective scratchpad: {:.1} MB", 0.52 * pad_mult as f64);
    println!("Effective iterations: {} per hash", 67968 * pad_mult);
    println!("Method: Run V3 hash {} times per 'hash' to simulate {}x iterations", pad_mult, pad_mult);
    println!();

    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let inputs: Vec<[u8; INPUT_SIZE]> = (0..hashes * max_cores).map(|_| {
        let mut input = [0u8; INPUT_SIZE]; rng.fill_bytes(&mut input); input
    }).collect();

    let core_counts: Vec<usize> = [1, 4, 8, 16, 32, max_cores]
        .iter().copied().filter(|&c| c <= max_cores)
        .collect::<std::collections::BTreeSet<usize>>().into_iter().collect();

    println!("{:>6} | {:>10} | {:>10}", "Cores", "H/s", "ms/hash");
    println!("{}", "-".repeat(40));

    for &cores in &core_counts {
        let total = hashes * cores;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(cores).build().unwrap();

        let start = Instant::now();
        pool.install(|| inputs[..total].par_iter().for_each(|input| {
            let mut sp = ScratchPad::default();
            // Run V3 pad_mult times to simulate larger scratchpad
            for p in 0..pad_mult {
                let mut modified_input = *input;
                modified_input[0] = modified_input[0].wrapping_add(p as u8);
                std::hint::black_box(v3::xelis_hash_v3(&modified_input, &mut sp));
            }
        }));
        let elapsed = start.elapsed();
        let hs = total as f64 / elapsed.as_secs_f64();
        let ms_per = elapsed.as_secs_f64() * 1000.0 / total as f64;

        println!("{:>6} | {:>10.1} | {:>10.1}", cores, hs, ms_per);
    }
}
