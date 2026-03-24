use blake3::hash as blake3_hash;
use chacha20::{
    cipher::{KeyIvInit, StreamCipher},
    ChaCha8,
};

pub const HASH_SIZE: usize = 32;
pub type Hash = [u8; HASH_SIZE];

// V3/V4 shared constants
pub const MEMORY_SIZE: usize = 531 * 128; // 67,968 u64s = ~544 KB
pub const MEMORY_SIZE_BYTES: usize = MEMORY_SIZE * 8;
pub const SCRATCHPAD_ITERS: usize = 2;
pub const BUFFER_SIZE: usize = MEMORY_SIZE / 2; // 33,984

const CHUNK_SIZE: usize = 32;
const NONCE_SIZE: usize = 12;

// ── Scratchpad ──────────────────────────────────────────────────────────────

pub struct ScratchPad {
    data: Box<[u64; MEMORY_SIZE]>,
}

impl ScratchPad {
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u64; MEMORY_SIZE] {
        &mut self.data
    }

    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(self.data.as_mut_slice())
    }
}

impl Default for ScratchPad {
    fn default() -> Self {
        Self {
            data: vec![0u64; MEMORY_SIZE]
                .into_boxed_slice()
                .try_into()
                .expect("Failed generating scratchpad"),
        }
    }
}

// ── Stage 1: ChaCha8 scratchpad fill (identical for V3 and V4) ─────────────

pub fn stage_1(input: &[u8], scratch_pad: &mut ScratchPad) {
    let bytes = scratch_pad.as_mut_bytes();
    bytes.fill(0);

    let mut output_offset = 0;
    let mut nonce = [0u8; NONCE_SIZE];
    let mut input_hash: [u8; 32] = blake3_hash(input).into();
    nonce.copy_from_slice(&input_hash[..NONCE_SIZE]);

    let num_chunks = (input.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (chunk_index, chunk) in input.chunks(CHUNK_SIZE).enumerate() {
        let mut tmp = [0u8; HASH_SIZE * 2];
        tmp[0..HASH_SIZE].copy_from_slice(&input_hash);
        tmp[HASH_SIZE..HASH_SIZE + chunk.len()].copy_from_slice(chunk);
        input_hash = blake3_hash(&tmp).into();

        let mut cipher = ChaCha8::new(&input_hash.into(), &nonce.into());

        let remaining_output_size = MEMORY_SIZE_BYTES - output_offset;
        let chunks_left = num_chunks - chunk_index;
        let chunk_output_size = remaining_output_size / chunks_left;
        let current_output_size = remaining_output_size.min(chunk_output_size);

        let offset = chunk_index * current_output_size;
        let part = &mut bytes[offset..offset + current_output_size];
        cipher.apply_keystream(part);

        output_offset += current_output_size;

        let nonce_start = current_output_size.saturating_sub(NONCE_SIZE);
        nonce.copy_from_slice(&part[nonce_start..]);
    }
}

// ── Stage 4: Blake3 hash of entire scratchpad (identical for V3 and V4) ─────

pub fn stage_4(scratch_pad: &[u64; MEMORY_SIZE]) -> Hash {
    let bytes: &[u8] = bytemuck::cast_slice(scratch_pad.as_slice());
    blake3_hash(bytes).into()
}

// ── Shared helper functions ─────────────────────────────────────────────────

#[inline(always)]
pub fn combine_u64(high: u64, low: u64) -> u128 {
    (high as u128) << 64 | low as u128
}

#[inline]
pub const fn murmurhash3(mut seed: u64) -> u64 {
    seed ^= seed >> 55;
    seed = seed.wrapping_mul(0xff51afd7ed558ccd);
    seed ^= seed >> 32;
    seed = seed.wrapping_mul(0xc4ceb9fe1a85ec53);
    seed ^= seed >> 15;
    seed
}

#[inline(always)]
pub fn map_index(mut x: u64) -> usize {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    ((x as u128) * (BUFFER_SIZE as u128) >> 64) as usize
}

#[inline(always)]
pub fn pick_half(seed: u64) -> bool {
    (murmurhash3(seed) & (1u64 << 58)) != 0
}

#[inline(always)]
pub fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let approx = (n as f64).sqrt() as u64;
    if approx.wrapping_mul(approx) > n {
        approx - 1
    } else if (approx + 1).wrapping_mul(approx + 1) <= n {
        approx + 1
    } else {
        approx
    }
}

pub const fn modular_power(mut base: u64, mut exp: u64, mod_: u64) -> u64 {
    let mut result: u64 = 1;
    base %= mod_;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % mod_ as u128) as u64;
        }
        base = ((base as u128 * base as u128) % mod_ as u128) as u64;
        exp /= 2;
    }
    result
}
