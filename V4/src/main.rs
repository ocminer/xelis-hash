mod common;
mod v3;
mod v4;

use common::ScratchPad;
use rand::{rngs::OsRng, RngCore};
use std::time::{Duration, Instant};

const INPUT_SIZE: usize = 112;

fn format_duration(d: Duration) -> String {
    if d.as_secs() > 0 {
        format!("{:.2}s", d.as_secs_f64())
    } else if d.as_millis() > 0 {
        format!("{:.1}ms", d.as_secs_f64() * 1000.0)
    } else {
        format!("{:.1}us", d.as_secs_f64() * 1_000_000.0)
    }
}

fn main() {
    let num_hashes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("XelisHash V3 vs V4 (S-box + MulChain) — CPU Benchmark");
    println!("======================================================");
    println!("Hashes per run:  {}", num_hashes);
    println!(
        "Scratchpad: {} KB, Inner: {} x {} = {}",
        common::MEMORY_SIZE * 8 / 1024,
        common::SCRATCHPAD_ITERS,
        common::BUFFER_SIZE,
        common::SCRATCHPAD_ITERS * common::BUFFER_SIZE
    );
    println!();

    // Random inputs
    let mut inputs: Vec<[u8; INPUT_SIZE]> = Vec::with_capacity(num_hashes);
    for _ in 0..num_hashes {
        let mut input = [0u8; INPUT_SIZE];
        OsRng.fill_bytes(&mut input);
        inputs.push(input);
    }

    // V3 test vector
    {
        let mut sp = ScratchPad::default();
        let hash = v3::xelis_hash_v3(&[0u8; INPUT_SIZE], &mut sp);
        let expected = [
            105, 172, 103, 40, 94, 253, 92, 162, 42, 252, 5, 196, 236, 238, 91, 218, 22, 157,
            228, 233, 239, 8, 250, 57, 212, 166, 121, 132, 148, 205, 103, 163,
        ];
        assert_eq!(hash, expected, "V3 test vector MISMATCH");
        println!("[PASS] V3 test vector verified");
    }

    // V4 determinism
    {
        let mut sp = ScratchPad::default();
        let h1 = v4::xelis_hash_v4(&[0u8; INPUT_SIZE], &mut sp);
        let h2 = v4::xelis_hash_v4(&[0u8; INPUT_SIZE], &mut sp);
        assert_eq!(h1, h2, "V4 not deterministic");
        assert_ne!(h1, [0u8; 32], "V4 zero output");
        println!(
            "[PASS] V4 deterministic ({:02x}{:02x}{:02x}{:02x}...)",
            h1[0], h1[1], h1[2], h1[3]
        );
    }

    // V3 != V4
    {
        let mut sp = ScratchPad::default();
        let h3 = v3::xelis_hash_v3(&[0u8; INPUT_SIZE], &mut sp);
        let h4 = v4::xelis_hash_v4(&[0u8; INPUT_SIZE], &mut sp);
        assert_ne!(h3, h4);
        println!("[PASS] V3 != V4");
    }
    println!();

    // Warmup
    {
        let mut sp = ScratchPad::default();
        for i in 0..3.min(num_hashes) {
            std::hint::black_box(v3::xelis_hash_v3(&inputs[i], &mut sp));
            std::hint::black_box(v4::xelis_hash_v4(&inputs[i], &mut sp));
        }
    }

    // Benchmark V3
    let mut sp = ScratchPad::default();
    let start = Instant::now();
    for input in &inputs {
        std::hint::black_box(v3::xelis_hash_v3(input, &mut sp));
    }
    let v3_total = start.elapsed();

    // Benchmark V4
    let start = Instant::now();
    for input in &inputs {
        std::hint::black_box(v4::xelis_hash_v4(input, &mut sp));
    }
    let v4_total = start.elapsed();

    let v3_hs = num_hashes as f64 / v3_total.as_secs_f64();
    let v4_hs = num_hashes as f64 / v4_total.as_secs_f64();
    let v3_per = v3_total / num_hashes as u32;
    let v4_per = v4_total / num_hashes as u32;

    println!("--- Full Hash ({} hashes, random inputs) ---", num_hashes);
    println!();
    println!(
        "  V3:  {} total, {} per hash, {:.2} H/s",
        format_duration(v3_total), format_duration(v3_per), v3_hs
    );
    println!(
        "  V4:  {} total, {} per hash, {:.2} H/s",
        format_duration(v4_total), format_duration(v4_per), v4_hs
    );
    println!(
        "  Ratio: V4 is {:.2}x slower ({:.1}% overhead)",
        v4_total.as_secs_f64() / v3_total.as_secs_f64(),
        (v4_total.as_secs_f64() / v3_total.as_secs_f64() - 1.0) * 100.0
    );
    println!();

    // Stage breakdown
    println!("--- Stage 3 Breakdown ({} hashes) ---", num_hashes);
    let mut s3_v3 = Duration::ZERO;
    let mut s3_v4 = Duration::ZERO;
    let mut s1_total = Duration::ZERO;

    for input in &inputs {
        let t = Instant::now();
        common::stage_1(input, &mut sp);
        s1_total += t.elapsed();

        let saved: Vec<u64> = sp.as_mut_slice().to_vec();

        let t = Instant::now();
        v3::stage_3(sp.as_mut_slice());
        s3_v3 += t.elapsed();

        sp.as_mut_slice().copy_from_slice(&saved);

        let t = Instant::now();
        v4::stage_3(sp.as_mut_slice());
        s3_v4 += t.elapsed();
    }

    let s1_avg = s1_total / num_hashes as u32;
    let s3_v3_avg = s3_v3 / num_hashes as u32;
    let s3_v4_avg = s3_v4 / num_hashes as u32;

    println!("  Stage 1 (ChaCha8):     {}", format_duration(s1_avg));
    println!("  Stage 3 V3:            {}", format_duration(s3_v3_avg));
    println!("  Stage 3 V4 (sbox+mul): {}", format_duration(s3_v4_avg));
    println!(
        "  Stage 3 overhead: {:.2}x ({:.1}%)",
        s3_v4_avg.as_secs_f64() / s3_v3_avg.as_secs_f64(),
        (s3_v4_avg.as_secs_f64() / s3_v3_avg.as_secs_f64() - 1.0) * 100.0
    );
    println!();

    // Mul chain micro-benchmark
    println!("--- Multiply Chain Micro-Benchmark ---");
    let iters: u64 = 10_000_000;
    let mut dummy: u64 = 0x123456789abcdef0;
    let start = Instant::now();
    for i in 0..iters {
        dummy = v4::compute_barrier_pub(dummy, i, dummy ^ i, i.wrapping_mul(3), dummy.rotate_left(7));
    }
    let chain_total = start.elapsed();
    std::hint::black_box(dummy);
    let ns_per = chain_total.as_nanos() as f64 / iters as f64;
    println!("  {:.1} ns/call ({:.0} cycles at 3.5 GHz)", ns_per, ns_per * 3.5);
    println!();

    // Summary
    println!("======================================================");
    println!("RESULTS SUMMARY");
    println!("======================================================");
    println!("V3: {:.2} H/s ({} per hash)", v3_hs, format_duration(v3_per));
    println!("V4: {:.2} H/s ({} per hash)", v4_hs, format_duration(v4_per));
    println!(
        "CPU overhead: {:.1}% (V4 stage3 is {:.2}x slower)",
        (v4_total.as_secs_f64() / v3_total.as_secs_f64() - 1.0) * 100.0,
        s3_v4_avg.as_secs_f64() / s3_v3_avg.as_secs_f64()
    );
    println!();
    println!("GPU comparison (from CUDA bench):");
    println!("  RTX 5090 V3: 27.0 kH/s -> V4: 11.7 kH/s (2.31x slower)");
    println!(
        "  CPU V3: {:.0} H/s -> V4: {:.0} H/s ({:.2}x slower)",
        v3_hs, v4_hs, v4_total.as_secs_f64() / v3_total.as_secs_f64()
    );
    println!(
        "  GPU/CPU ratio V3: {:.0}x -> V4: {:.0}x",
        27000.0 / v3_hs,
        11700.0 / v4_hs
    );
}
