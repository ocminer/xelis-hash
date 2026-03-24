/// Full V4 CPU benchmark: real V3 hash + shared SUPRNOVA dataset reads.
/// Dataset is 256MB shared across all threads → DRAM bandwidth caps scaling.
mod common;
mod v3;

use common::{ScratchPad, BUFFER_SIZE, SCRATCHPAD_ITERS};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const INPUT_SIZE: usize = 112;
const SUPRNOVA_SEED: u64 = 0x5355505249564F41; // "SUPRNOVA" as u64

const fn parse_env(s: &str, _d: usize) -> usize {
    let b = s.as_bytes();
    let mut n = 0usize;
    let mut i = 0;
    while i < b.len() { n = n * 10 + (b[i] - b'0') as usize; i += 1; }
    n
}

const DATASET_SIZE_MB: usize = match option_env!("DATASET_SIZE_MB") {
    Some(s) => parse_env(s, 256), None => 256
};
const DATASET_SIZE: usize = DATASET_SIZE_MB * 1024 * 1024 / 8; // in u64s
const DATASET_READS: usize = match option_env!("DATASET_READS") {
    Some(s) => parse_env(s, 2), None => 2
};
const TAPE_LEN: usize = match option_env!("TAPE_LEN") {
    Some(s) => parse_env(s, 32), None => 32
};
const TAPE_FREQ: usize = match option_env!("TAPE_FREQ") {
    Some(s) => parse_env(s, 32), None => 32
};

#[inline(always)]
fn map_dataset(x: u64) -> usize {
    let x = x ^ (x >> 33);
    let x = x.wrapping_mul(0xff51afd7ed558ccdu64);
    let xhi = (x >> 32) as u32;
    let xlo = x as u32;
    let carry = ((xlo as u64).wrapping_mul(DATASET_SIZE as u64) >> 32) as u32;
    (((xhi as u64).wrapping_mul(DATASET_SIZE as u64).wrapping_add(carry as u64)) >> 32) as usize
}

/// Simulates tape execution overhead on CPU.
/// Branch predictor learns the fixed tape → near-zero overhead after first pass.
#[inline(never)]
fn tape_exec(opcode: u8, s0: u64, s1: u64, operand: u64) -> u64 {
    match opcode & 0xF {
        0 => s0 ^ operand,
        1 => s0.wrapping_add(operand),
        2 => s0.rotate_left((operand & 63) as u32),
        3 => s0.wrapping_mul(operand | 1),
        4 => s0 ^ s1.rotate_left(17) ^ operand,
        5 => s0.wrapping_add(s1).wrapping_mul(operand | 1),
        6 => s0 ^ common::murmurhash3(operand),
        7 => s0.wrapping_mul(s1).rotate_left((operand & 63) as u32) ^ operand,
        8 => s0 ^ (s1 >> (operand as u32 & 31)),
        9 => (s0 & operand) | (s1 & !operand),
        10 => common::murmurhash3(s0 ^ s1).wrapping_add(operand),
        11 => { ((s0 as u128).wrapping_mul(operand as u128) >> 64) as u64 ^ s1 }
        12 => common::isqrt(s0 ^ operand),
        13 => s0 ^ s1.wrapping_mul(operand | 1), // simplified AES stand-in (CPU has AES-NI)
        14 => common::modular_power(s0 & 0xFFFF, s1 & 0x1F, operand | 3),
        15 => {
            let n = ((s0 as u128) << 64) | (s1 as u128);
            (n % ((operand | 1) as u128)) as u64
        }
        _ => s0,
    }
}

/// Full V4 hash: real V3 + dataset reads + S-box + tape
fn v4_full_hash(input: &[u8; INPUT_SIZE], sp: &mut ScratchPad, dataset: &[u64]) {
    // Run real V3 hash (stage 1 + stage 3 + stage 4)
    // But intercept to add dataset reads + tape
    common::stage_1(input, sp);

    let scratch = sp.as_mut_slice();
    let (mem_a, mem_b) = scratch.split_at_mut(BUFFER_SIZE);

    // Generate tape from scratchpad
    let mut tape = [0u8; 64]; // max TAPE_LEN
    for k in 0..TAPE_LEN.min(64) {
        tape[k] = ((mem_a[k * (BUFFER_SIZE / TAPE_LEN.max(1))] >> 56) & 0xF) as u8;
    }

    // S-box (L1-resident on CPU, ~free)
    let mut sbox = [0u64; 256];
    for k in 0..256 {
        sbox[k] = mem_a[k] ^ (0xA5A5A5A5A5A5A5A5u64.wrapping_add(k as u64));
    }

    let mut addr_a = mem_b[BUFFER_SIZE - 1];
    let mut addr_b = mem_a[BUFFER_SIZE - 1] >> 32;
    let mut r: usize = 0;
    let mut tape_state: u64 = addr_a ^ addr_b;

    let key = aes::cipher::generic_array::GenericArray::from(*b"xelishash-pow-v4");
    let mut block = aes::cipher::generic_array::GenericArray::from([0u8; 16]);

    for i in 0..SCRATCHPAD_ITERS {
        let ia = common::map_index(addr_a);
        let mem_a_val = mem_a[ia];
        let ib = common::map_index(mem_a_val ^ addr_b);
        let mem_b_val = mem_b[ib];
        block[..8].copy_from_slice(&mem_b_val.to_le_bytes());
        block[8..].copy_from_slice(&mem_a_val.to_le_bytes());
        aes::hazmat::cipher_round(&mut block, &key);
        let h1 = u64::from_le_bytes(block[..8].try_into().unwrap());
        let h2 = u64::from_le_bytes(block[8..].try_into().unwrap());
        let mut result = !(h1 ^ h2);

        for j in 0..BUFFER_SIZE {
            let a = mem_a[common::map_index(result)];
            let b = mem_b[common::map_index(a ^ !result.rotate_right(r as u32))];
            let c = if r < BUFFER_SIZE { mem_a[r] } else { mem_b[r - BUFFER_SIZE] };
            r = if r < common::MEMORY_SIZE - 1 { r + 1 } else { 0 };

            // S-box (L1 on CPU)
            let mut chain = a ^ sbox[((a >> 56) & 0xFF) as usize];
            sbox[((chain >> 48) & 0xFF) as usize] ^= chain;

            // Dataset reads: random page, sequential within page.
            // Every 64 iters: pick new random 4KB page from 256MB dataset.
            // Within page: sequential reads → CPU prefetcher handles it (~5ns/read).
            // GPU: 32 threads pick 32 different pages → scattered → full VRAM latency.
            // At 32 CPU cores, each doing ~530 page-hops/hash → L3 contention on 256MB dataset.
            if DATASET_READS > 0 {
                let page_size = 512usize; // 512 u64s = 4KB page
                let page_idx = if (j & 63) == 0 {
                    // New random page every 64 iters
                    map_dataset(result ^ chain) / page_size * page_size
                } else {
                    // Continue within current page (sequential)
                    ((addr_a.wrapping_add(j as u64)) as usize) % DATASET_SIZE / page_size * page_size
                };
                let offset = (j as usize) & (page_size - 1);
                for dr in 0..DATASET_READS {
                    let didx = page_idx + ((offset + dr) % page_size);
                    chain ^= dataset[didx];
                }
                chain = chain.rotate_left(13);
            }

            // TapeMix (branch-predicted on CPU)
            if (j & (TAPE_FREQ - 1)) == 0 {
                for t in 0..TAPE_LEN {
                    tape_state = tape_exec(tape[t], tape_state, result ^ a, c ^ chain);
                }
                result ^= tape_state;
            }

            let sel = (result.rotate_left(c as u32) & 0xf) as u8;
            let v = match sel {
                0 => { let t1 = common::combine_u64(a.wrapping_add(i as u64), common::isqrt(b.wrapping_add(j as u64))); let d = common::murmurhash3(c ^ result ^ i as u64 ^ j as u64) | 1; (t1 % (d as u128)) as u64 }
                1 => { let sq = common::isqrt(b | 2); c.wrapping_add(i as u64).wrapping_rem(sq).rotate_left(i.wrapping_add(j) as u32).wrapping_mul(common::isqrt(a.wrapping_add(j as u64))) }
                2 => common::isqrt(a.wrapping_add(i as u64)).wrapping_mul(common::isqrt(c.wrapping_add(j as u64))) ^ b.wrapping_add(i as u64).wrapping_add(j as u64),
                3 => a.wrapping_add(b).wrapping_mul(c),
                4 => b.wrapping_sub(c).wrapping_mul(a),
                5 => c.wrapping_sub(a).wrapping_add(b),
                6 => a.wrapping_sub(b).wrapping_add(c),
                7 => b.wrapping_mul(c).wrapping_add(a),
                8 => c.wrapping_mul(a).wrapping_add(b),
                9 => a.wrapping_mul(b).wrapping_mul(c),
                10 => (common::combine_u64(a, b).wrapping_rem((c | 1) as u128)) as u64,
                11 => { let t1 = common::combine_u64(b, c); let t2 = common::combine_u64(result.rotate_left(r as u32), a | 2); if t2 > t1 { c } else { t1.wrapping_rem(t2) as u64 } }
                12 => (common::combine_u64(c, a).wrapping_div((b | 4) as u128)) as u64,
                13 => { let t1 = common::combine_u64(result.rotate_left(r as u32), b); let t2 = common::combine_u64(a, c | 8); if t1 > t2 { t1.wrapping_div(t2) as u64 } else { a ^ b } }
                14 => (common::combine_u64(b, a).wrapping_mul(c as u128) >> 64) as u64,
                15 => (common::combine_u64(a, c).wrapping_mul(common::combine_u64(result.rotate_right(r as u32), b)) >> 64) as u64,
                _ => unreachable!(),
            };

            let idx_seed = (v ^ chain) ^ result;
            result = idx_seed.rotate_left(r as u32);
            let use_b = common::pick_half(v);
            let it = common::map_index(idx_seed);
            let t = if use_b { mem_b[it] } else { mem_a[it] } ^ result;
            let wa = common::map_index(t ^ result ^ 0x9e3779b97f4a7c15);
            let wb = common::map_index(wa as u64 ^ !result ^ 0xd2b74407b1ce6e93);
            let old_a = std::mem::replace(&mut mem_a[wa], t);
            mem_b[wb] ^= old_a ^ t.rotate_right(i.wrapping_add(j) as u32);
        }
        addr_a = common::modular_power(addr_a, addr_b, result);
        addr_b = common::isqrt(result).wrapping_mul((r as u64).wrapping_add(1)).wrapping_mul(common::isqrt(addr_a));
    }
    // Prevent DCE
    std::hint::black_box(mem_a[0] ^ addr_a ^ addr_b ^ tape_state);
}

fn generate_dataset() -> Vec<u64> {
    println!("Generating {} MB SUPRNOVA dataset (0x{:016X})...", DATASET_SIZE_MB, SUPRNOVA_SEED);
    let mut dataset = vec![0u64; DATASET_SIZE];
    for i in 0..DATASET_SIZE {
        let mut s = SUPRNOVA_SEED ^ ((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        s ^= s >> 30; s = s.wrapping_mul(0xBF58476D1CE4E5B9);
        s ^= s >> 27; s = s.wrapping_mul(0x94D049BB133111EB);
        s ^= s >> 31;
        if (i & 0xFFF) == 0x53 { s ^= SUPRNOVA_SEED; }
        dataset[i] = s;
    }
    println!("Done. dataset[0x53] = 0x{:016X}", dataset[0x53]);
    dataset
}

fn main() {
    let hashes: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let max_cores: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(num_cpus::get_physical());

    println!("================================================================");
    println!("XelisHash V4 Full CPU — S-box + Tape + Dataset (REAL V3 work)");
    println!("================================================================");
    println!("Dataset: {} MB, {} reads/iter", DATASET_SIZE_MB, DATASET_READS);
    println!("Tape: len={}, freq={}", TAPE_LEN, TAPE_FREQ);
    println!("Hashes per core: {}", hashes);
    println!();

    let dataset = Arc::new(generate_dataset());

    // Also bench plain V3 for reference
    {
        let mut sp = ScratchPad::default();
        let input = [0u8; INPUT_SIZE];
        let t = Instant::now();
        for _ in 0..5 { std::hint::black_box(v3::xelis_hash_v3(&input, &mut sp)); }
        let v3_hs = 5.0 / t.elapsed().as_secs_f64();
        println!("V3 reference (1 core): {:.1} H/s ({:.1} ms/hash)\n", v3_hs, 1000.0/v3_hs);
    }

    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let inputs: Vec<[u8; INPUT_SIZE]> = (0..hashes * max_cores).map(|_| {
        let mut input = [0u8; INPUT_SIZE]; rng.fill_bytes(&mut input); input
    }).collect();

    let core_counts: Vec<usize> = [1, 4, 8, 16, 32, max_cores]
        .iter().copied().filter(|&c| c <= max_cores)
        .collect::<std::collections::BTreeSet<usize>>().into_iter().collect();

    println!("{:>6} | {:>10} | {:>8} | {:>8}", "Cores", "H/s", "ms/hash", "scale");
    println!("{}", "-".repeat(46));

    let mut base_hs = 0.0f64;
    for &cores in &core_counts {
        let total = hashes * cores;
        let ds = Arc::clone(&dataset);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(cores).build().unwrap();
        let start = Instant::now();
        pool.install(|| {
            inputs[..total].par_iter().for_each(|input| {
                let mut sp = ScratchPad::default();
                v4_full_hash(input, &mut sp, &ds);
            });
        });
        let elapsed = start.elapsed();
        let hs = total as f64 / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / total as f64;
        if base_hs == 0.0 { base_hs = hs; }
        let scale = hs / base_hs;
        println!("{:>6} | {:>10.1} | {:>8.1} | {:>7.1}x", cores, hs, ms, scale);
    }
}
