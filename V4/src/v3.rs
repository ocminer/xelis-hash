/// XelisHashV3 — exact copy of the reference implementation for benchmarking.
/// This is the baseline we compare V4 against.
use aes::cipher::generic_array::GenericArray;
use crate::common::*;

const KEY: [u8; 16] = *b"xelishash-pow-v3";

pub fn stage_3(scratch_pad: &mut [u64; MEMORY_SIZE]) {
    let key = GenericArray::from(KEY);
    let mut block = GenericArray::from([0u8; 16]);

    let (mem_buffer_a, mem_buffer_b) = scratch_pad.as_mut_slice().split_at_mut(BUFFER_SIZE);

    let mut addr_a = mem_buffer_b[BUFFER_SIZE - 1];
    let mut addr_b = mem_buffer_a[BUFFER_SIZE - 1] >> 32;
    let mut r: usize = 0;

    for i in 0..SCRATCHPAD_ITERS {
        let index_a = map_index(addr_a);
        let mem_a = mem_buffer_a[index_a];
        let index_b = map_index(mem_a ^ addr_b);
        let mem_b = mem_buffer_b[index_b];

        block[..8].copy_from_slice(&mem_b.to_le_bytes());
        block[8..].copy_from_slice(&mem_a.to_le_bytes());

        aes::hazmat::cipher_round(&mut block, &key);

        let hash1 = u64::from_le_bytes(block[..8].try_into().unwrap());
        let hash2 = u64::from_le_bytes(block[8..].try_into().unwrap());
        let mut result = !(hash1 ^ hash2);

        for j in 0..BUFFER_SIZE {
            let index_a = map_index(result);
            let a = mem_buffer_a[index_a];
            let index_b = map_index(a ^ !result.rotate_right(r as u32));
            let b = mem_buffer_b[index_b];

            let c = if r < BUFFER_SIZE {
                mem_buffer_a[r]
            } else {
                mem_buffer_b[r - BUFFER_SIZE]
            };
            r = if r < MEMORY_SIZE - 1 { r + 1 } else { 0 };

            let branch_idx = (result.rotate_left(c as u32) & 0xf) as u8;

            let v = match branch_idx {
                0 => {
                    let t1 = combine_u64(
                        a.wrapping_add(i as u64),
                        isqrt(b.wrapping_add(j as u64)),
                    );
                    let denom = murmurhash3(c ^ result ^ i as u64 ^ j as u64) | 1;
                    (t1 % (denom as u128)) as u64
                }
                1 => {
                    let t1 = c.wrapping_add(i as u64).wrapping_rem(isqrt(b | 2));
                    let t2 = t1.rotate_left((i.wrapping_add(j)) as u32);
                    t2.wrapping_mul(isqrt(a.wrapping_add(j as u64)))
                }
                2 => {
                    let t1 = isqrt(a.wrapping_add(i as u64));
                    let t2 = isqrt(c.wrapping_add(j as u64));
                    t1.wrapping_mul(t2) ^ b.wrapping_add(i as u64).wrapping_add(j as u64)
                }
                3 => a.wrapping_add(b).wrapping_mul(c),
                4 => b.wrapping_sub(c).wrapping_mul(a),
                5 => c.wrapping_sub(a).wrapping_add(b),
                6 => a.wrapping_sub(b).wrapping_add(c),
                7 => b.wrapping_mul(c).wrapping_add(a),
                8 => c.wrapping_mul(a).wrapping_add(b),
                9 => a.wrapping_mul(b).wrapping_mul(c),
                10 => {
                    let t1 = combine_u64(a, b);
                    let t2 = (c | 1) as u128;
                    t1.wrapping_rem(t2) as u64
                }
                11 => {
                    let t1 = combine_u64(b, c);
                    let t2 = combine_u64(result.rotate_left(r as u32), a | 2);
                    if t2 > t1 { c } else { t1.wrapping_rem(t2) as u64 }
                }
                12 => {
                    let t1 = combine_u64(c, a);
                    let t2 = (b | 4) as u128;
                    t1.wrapping_div(t2) as u64
                }
                13 => {
                    let t1 = combine_u64(result.rotate_left(r as u32), b);
                    let t2 = combine_u64(a, c | 8);
                    if t1 > t2 {
                        t1.wrapping_div(t2) as u64
                    } else {
                        a ^ b
                    }
                }
                14 => {
                    let t1 = combine_u64(b, a);
                    let t2 = c as u128;
                    (t1.wrapping_mul(t2) >> 64) as u64
                }
                15 => {
                    let t1 = combine_u64(a, c);
                    let t2 = combine_u64(result.rotate_right(r as u32), b);
                    (t1.wrapping_mul(t2) >> 64) as u64
                }
                _ => unreachable!(),
            };

            let seed = v ^ result;
            result = seed.rotate_left(r as u32);

            let use_buffer_b = pick_half(v);
            let index_t = map_index(seed);
            let t = if use_buffer_b {
                mem_buffer_b[index_t]
            } else {
                mem_buffer_a[index_t]
            } ^ result;

            let index_a = map_index(t ^ result ^ 0x9e3779b97f4a7c15);
            let index_b = map_index(index_a as u64 ^ !result ^ 0xd2b74407b1ce6e93);

            let old_a = std::mem::replace(&mut mem_buffer_a[index_a], t);
            mem_buffer_b[index_b] ^= old_a ^ t.rotate_right(i.wrapping_add(j) as u32);
        }

        addr_a = modular_power(addr_a, addr_b, result);
        addr_b = isqrt(result)
            .wrapping_mul((r as u64).wrapping_add(1))
            .wrapping_mul(isqrt(addr_a));
    }
}

pub fn xelis_hash_v3(input: &[u8], scratch_pad: &mut ScratchPad) -> Hash {
    stage_1(input, scratch_pad);
    let sp = scratch_pad.as_mut_slice();
    stage_3(sp);
    stage_4(sp)
}
