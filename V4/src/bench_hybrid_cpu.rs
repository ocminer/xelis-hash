/// CPU benchmark for BigPad + AES hybrid.
/// AES-NI gives CPU ~1 cycle/round. GPU pays ~20 cycles/round.
/// PAD_MULT simulated by running V3 N times (same cache behavior).
/// AES_MIX adds real AES rounds per "iteration" via aes crate (uses AES-NI).
mod common;
mod v3;

use aes::cipher::generic_array::GenericArray;
use common::ScratchPad;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const INPUT_SIZE: usize = 112;

const fn parse_env(s: &str, _d: usize) -> usize {
    let b = s.as_bytes(); let mut n = 0usize; let mut i = 0;
    while i < b.len() { n = n * 10 + (b[i] - b'0') as usize; i += 1; } n
}
const PAD_MULT: usize = match option_env!("PAD_MULT") { Some(s) => parse_env(s, 6), None => 6 };
const AES_MIX: usize = match option_env!("AES_MIX") { Some(s) => parse_env(s, 0), None => 0 };

/// Simulate one "big hash" = PAD_MULT V3 hashes + AES overhead.
/// The AES overhead is measured separately and added per-iteration.
fn hybrid_hash(input: &[u8; INPUT_SIZE], sp: &mut ScratchPad) {
    for p in 0..PAD_MULT {
        let mut modified = *input;
        modified[0] = modified[0].wrapping_add(p as u8);
        std::hint::black_box(v3::xelis_hash_v3(&modified, sp));
    }
    // AES mixing overhead: AES_MIX rounds per inner iteration, total PAD_MULT * 67968 iters.
    // Each AES round on CPU with AES-NI: ~1 cycle = ~0.3ns at 3.5 GHz.
    // Simulate by running actual AES rounds.
    if AES_MIX > 0 {
        let total_aes_iters = PAD_MULT * 67968;
        let key = GenericArray::from([0x42u8; 16]);
        let mut block = GenericArray::from([0u8; 16]);
        // Copy some state into block to prevent optimization
        let sp_slice = sp.as_mut_slice();
        block[..8].copy_from_slice(&sp_slice[0].to_le_bytes());
        block[8..].copy_from_slice(&sp_slice[1].to_le_bytes());

        for _ in 0..total_aes_iters {
            for _ in 0..AES_MIX {
                aes::hazmat::cipher_round(&mut block, &key);
            }
        }
        // Write back to prevent dead-code elimination
        let v = u64::from_le_bytes(block[..8].try_into().unwrap());
        sp_slice[0] ^= std::hint::black_box(v);
    }
}

fn main() {
    let hashes: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let max_cores: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(num_cpus::get_physical());

    println!("================================================================");
    println!("Hybrid CPU Benchmark — PAD_MULT={}, AES_MIX={}", PAD_MULT, AES_MIX);
    println!("================================================================");
    println!("Effective scratchpad: {:.1} MB", 0.52 * PAD_MULT as f64);
    println!("AES rounds per inner iter: {} (using AES-NI)", AES_MIX);
    println!("Total AES rounds per hash: {}", PAD_MULT * 67968 * AES_MIX);
    println!();

    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let inputs: Vec<[u8; INPUT_SIZE]> = (0..hashes * max_cores).map(|_| {
        let mut input = [0u8; INPUT_SIZE]; rng.fill_bytes(&mut input); input
    }).collect();

    let core_counts: Vec<usize> = [1, 8, 16, 32, max_cores]
        .iter().copied().filter(|&c| c <= max_cores)
        .collect::<std::collections::BTreeSet<usize>>().into_iter().collect();

    println!("{:>6} | {:>10} | {:>8}", "Cores", "H/s", "ms/hash");
    println!("{}", "-".repeat(36));

    for &cores in &core_counts {
        let total = hashes * cores;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(cores).build().unwrap();
        let start = Instant::now();
        pool.install(|| inputs[..total].par_iter().for_each(|input| {
            let mut sp = ScratchPad::default();
            hybrid_hash(input, &mut sp);
        }));
        let elapsed = start.elapsed();
        let hs = total as f64 / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / total as f64;
        println!("{:>6} | {:>10.1} | {:>8.1}", cores, hs, ms);
    }
}
