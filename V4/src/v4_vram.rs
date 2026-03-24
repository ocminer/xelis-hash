/// XelisHashV4-VRAM — Large per-thread V-array (Argon2d-style)
/// VARRAY_SIZE and VARRAY_READS controlled by env vars at compile time.
use aes::cipher::generic_array::GenericArray;
use crate::common::*;

const KEY: [u8; 16] = *b"xelishash-pow-v4";
const STATE_WORDS: usize = 8;

const fn parse_env(s: &str, _default: usize) -> usize {
    let b = s.as_bytes();
    let mut n = 0usize;
    let mut i = 0;
    while i < b.len() { n = n * 10 + (b[i] - b'0') as usize; i += 1; }
    n
}

pub const VARRAY_SIZE: usize = match option_env!("VARRAY_SIZE") {
    Some(s) => parse_env(s, 262144),
    None => 262144,  // 2MB default
};
const VARRAY_READS: usize = match option_env!("VARRAY_READS") {
    Some(s) => parse_env(s, 2),
    None => 2,
};

#[inline(always)]
// Local murmurhash3 for V-array fill (matches CUDA kernel)
fn mmh3(mut s: u64) -> u64 {
    s ^= s >> 55; s = s.wrapping_mul(0xff51afd7ed558ccdu64);
    s ^= s >> 32; s = s.wrapping_mul(0xc4ceb9fe1a85ec53u64);
    s ^= s >> 15; s
}

#[inline(always)]
fn map_varray(x: u64) -> usize {
    let x = x ^ (x >> 33);
    let x = x.wrapping_mul(0xff51afd7ed558ccdu64);
    let xhi = (x >> 32) as u32;
    let xlo = x as u32;
    let carry = ((xlo as u64).wrapping_mul(VARRAY_SIZE as u64) >> 32) as u32;
    (((xhi as u64).wrapping_mul(VARRAY_SIZE as u64).wrapping_add(carry as u64)) >> 32) as usize
}

pub fn stage_3(scratch_pad: &mut [u64; MEMORY_SIZE], varray: &mut Vec<u64>) {
    let key = GenericArray::from(KEY);
    let mut block = GenericArray::from([0u8; 16]);
    let (mem_buffer_a, mem_buffer_b) = scratch_pad.as_mut_slice().split_at_mut(BUFFER_SIZE);

    // Fill V-array sequentially (Argon2d-style)
    varray.resize(VARRAY_SIZE, 0);
    {
        let mut prev = mem_buffer_a[0] ^ 0x6A09E667F3BCC908u64;
        for k in 0..VARRAY_SIZE {
            let sp_val = mem_buffer_a[k % BUFFER_SIZE];
            prev = prev ^ sp_val;
            prev = mmh3(prev);
            varray[k] = prev;
        }
    }

    let mut state = [0u64; STATE_WORDS];
    for k in 0..STATE_WORDS { state[k] = varray[k]; }

    let mut addr_a = mem_buffer_b[BUFFER_SIZE - 1];
    let mut addr_b = mem_buffer_a[BUFFER_SIZE - 1] >> 32;
    let mut r: usize = 0;

    for i in 0..SCRATCHPAD_ITERS {
        let ia = map_index(addr_a);
        let mem_a = mem_buffer_a[ia];
        let ib = map_index(mem_a ^ addr_b);
        let mem_b = mem_buffer_b[ib];
        block[..8].copy_from_slice(&mem_b.to_le_bytes());
        block[8..].copy_from_slice(&mem_a.to_le_bytes());
        aes::hazmat::cipher_round(&mut block, &key);
        let hash1 = u64::from_le_bytes(block[..8].try_into().unwrap());
        let hash2 = u64::from_le_bytes(block[8..].try_into().unwrap());
        let mut result = !(hash1 ^ hash2);

        for j in 0..BUFFER_SIZE {
            let ia = map_index(result);
            let a = mem_buffer_a[ia];
            let ib = map_index(a ^ !result.rotate_right(r as u32));
            let b = mem_buffer_b[ib];
            let c = if r < BUFFER_SIZE { mem_buffer_a[r] } else { mem_buffer_b[r-BUFFER_SIZE] };
            r = if r < MEMORY_SIZE-1 { r+1 } else { 0 };

            let sel = (result.rotate_left(c as u32) & 0xf) as u8;
            let v = match sel {
                0 => { let t1=combine_u64(a.wrapping_add(i as u64),isqrt(b.wrapping_add(j as u64))); let d=crate::common::murmurhash3(c^result^i as u64^j as u64)|1; (t1%(d as u128)) as u64 }
                1 => { let t1=c.wrapping_add(i as u64).wrapping_rem(isqrt(b|2)); t1.rotate_left((i.wrapping_add(j)) as u32).wrapping_mul(isqrt(a.wrapping_add(j as u64))) }
                2 => { isqrt(a.wrapping_add(i as u64)).wrapping_mul(isqrt(c.wrapping_add(j as u64))) ^ b.wrapping_add(i as u64).wrapping_add(j as u64) }
                3 => a.wrapping_add(b).wrapping_mul(c),
                4 => b.wrapping_sub(c).wrapping_mul(a),
                5 => c.wrapping_sub(a).wrapping_add(b),
                6 => a.wrapping_sub(b).wrapping_add(c),
                7 => b.wrapping_mul(c).wrapping_add(a),
                8 => c.wrapping_mul(a).wrapping_add(b),
                9 => a.wrapping_mul(b).wrapping_mul(c),
                10 => { (combine_u64(a,b).wrapping_rem((c|1) as u128)) as u64 }
                11 => { let t1=combine_u64(b,c); let t2=combine_u64(result.rotate_left(r as u32),a|2); if t2>t1{c}else{t1.wrapping_rem(t2) as u64} }
                12 => { (combine_u64(c,a).wrapping_div((b|4) as u128)) as u64 }
                13 => { let t1=combine_u64(result.rotate_left(r as u32),b); let t2=combine_u64(a,c|8); if t1>t2{t1.wrapping_div(t2) as u64}else{a^b} }
                14 => { (combine_u64(b,a).wrapping_mul(c as u128) >> 64) as u64 }
                15 => { (combine_u64(a,c).wrapping_mul(combine_u64(result.rotate_right(r as u32),b)) >> 64) as u64 }
                _ => unreachable!(),
            };

            // V4-VRAM: dependent V-array reads
            let mut chain = v;
            for m in 0..VARRAY_READS {
                let vidx = map_varray(chain ^ result ^ (m as u64).wrapping_mul(0x9E3779B97F4A7C15u64));
                chain ^= varray[vidx];
                chain = chain.rotate_left(17);
            }

            let si = state[((result >> 4) & (STATE_WORDS as u64 -1)) as usize];
            chain ^= si;
            state[((result >> 10) & (STATE_WORDS as u64 -1)) as usize] = chain;

            // V-array write-back
            let widx = map_varray(chain ^ 0xD2B74407B1CE6E93u64);
            varray[widx] ^= chain;

            let idx_seed = (v ^ chain) ^ result;
            result = idx_seed.rotate_left(r as u32);

            let use_b = pick_half(v);
            let it = map_index(idx_seed);
            let t = if use_b { mem_buffer_b[it] } else { mem_buffer_a[it] } ^ result;
            let wa = map_index(t ^ result ^ 0x9e3779b97f4a7c15);
            let wb = map_index(wa as u64 ^ !result ^ 0xd2b74407b1ce6e93);
            let old_a = std::mem::replace(&mut mem_buffer_a[wa], t);
            mem_buffer_b[wb] ^= old_a ^ t.rotate_right(i.wrapping_add(j) as u32);
        }
        addr_a = modular_power(addr_a, addr_b, result);
        addr_b = isqrt(result).wrapping_mul((r as u64).wrapping_add(1)).wrapping_mul(isqrt(addr_a));
    }
}

pub fn xelis_hash_v4_vram(input: &[u8], scratch_pad: &mut ScratchPad, varray: &mut Vec<u64>) -> Hash {
    stage_1(input, scratch_pad);
    let sp = scratch_pad.as_mut_slice();
    stage_3(sp, varray);
    stage_4(sp)
}
