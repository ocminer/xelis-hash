// xelis_v3v4_bench.cu — Standalone GPU benchmark: XelisHash V3 vs V4 stage_3
//
// Compiles with: nvcc -O3 -arch=sm_120 -o bench_v3v4 xelis_v3v4_bench.cu --use_fast_math
// Usage:         ./bench_v3v4 [device] [batch] [runs]
//
// Fills scratchpads with random data (curand), then times stage_3 for both
// V3 (original) and V4 (with Salsa20/8 compute barrier).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// ===========================================================================
// CONSTANTS
// ===========================================================================
#define MEMORY_SIZE   (531u * 128u)   // 67968 u64s = ~544 KB per hash
#define BUFFER_SIZE   (MEMORY_SIZE / 2u) // 33984
#define SCRATCHPAD_ITERS 2u
#define MEMORY_SIZE_BYTES (MEMORY_SIZE * 8u)

// ===========================================================================
// AES S-BOX (Rijndael)
// ===========================================================================
__constant__ uint8_t AES_SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

// ===========================================================================
// DEVICE HELPER FUNCTIONS
// ===========================================================================

__device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) {
    n &= 63u;
    return (x << n) | (x >> ((64u - n) & 63u));
}

__device__ __forceinline__ uint64_t rotr64(uint64_t x, uint32_t n) {
    n &= 63u;
    return (x >> n) | (x << ((64u - n) & 63u));
}

__device__ __forceinline__ uint64_t murmurhash3(uint64_t seed) {
    seed ^= seed >> 55;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 32;
    seed *= 0xc4ceb9fe1a85ec53ULL;
    seed ^= seed >> 15;
    return seed;
}

__device__ __forceinline__ uint64_t map_index(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    uint32_t xhi = (uint32_t)(x >> 32);
    uint32_t xlo = (uint32_t)x;
    uint32_t carry = (uint32_t)(((uint64_t)xlo * BUFFER_SIZE) >> 32);
    return ((uint64_t)xhi * BUFFER_SIZE + carry) >> 32;
}

__device__ __forceinline__ bool pick_half(uint64_t seed) {
    return (murmurhash3(seed) & (1ULL << 58)) != 0;
}

__device__ __forceinline__ uint64_t isqrt_v3(uint64_t n) {
    if (n < 2) return n;
    uint64_t x = __double2ull_rz(__dsqrt_rn(__ull2double_rn(n)));
    if (x > 4294967295ULL) x = 4294967295ULL;
    uint64_t sq = x * x;
    if (sq > n || (x > 0 && sq / x != x)) {
        x--;
        sq = x * x;
        if (sq > n) x--;
    } else {
        uint64_t gap = n - sq;
        uint64_t needed = 2 * x + 1;
        if (gap >= needed) {
            x++;
            sq = x * x;
            gap = n - sq;
            needed = 2 * x + 1;
            if (gap >= needed) x++;
        }
    }
    return x;
}

__device__ __forceinline__ uint64_t modular_power(uint64_t base, uint64_t exp, uint64_t mod) {
    if (mod == 0) return 0;
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            __uint128_t val = (__uint128_t)result * base;
            result = (uint64_t)(val % mod);
        }
        __uint128_t val = (__uint128_t)base * base;
        base = (uint64_t)(val % mod);
        exp >>= 1;
    }
    return result;
}

__device__ __forceinline__ __uint128_t combine_u64(uint64_t hi, uint64_t lo) {
    return ((__uint128_t)hi << 64) | lo;
}

__device__ __forceinline__ uint64_t fast_mulhi_128_64(uint64_t hi, uint64_t lo, uint64_t m) {
    return __umul64hi(lo, m) + hi * m;
}

__device__ __forceinline__ uint64_t fast_mulhi_256_128(uint64_t a, uint64_t c, uint64_t d, uint64_t b) {
    return __umul64hi(c, b) + a * b + c * d;
}

// ===========================================================================
// AES SINGLE ROUND
// ===========================================================================

__device__ __forceinline__ uint8_t gf_mul2(uint8_t x) {
    return (x << 1) ^ ((x >> 7) * 0x1b);
}

__device__ __forceinline__ void aes_single_round(uint8_t block[16], const uint8_t key[16]) {
    uint8_t s0  = AES_SBOX[block[0]];  uint8_t s1  = AES_SBOX[block[5]];
    uint8_t s2  = AES_SBOX[block[10]]; uint8_t s3  = AES_SBOX[block[15]];
    uint8_t s4  = AES_SBOX[block[4]];  uint8_t s5  = AES_SBOX[block[9]];
    uint8_t s6  = AES_SBOX[block[14]]; uint8_t s7  = AES_SBOX[block[3]];
    uint8_t s8  = AES_SBOX[block[8]];  uint8_t s9  = AES_SBOX[block[13]];
    uint8_t s10 = AES_SBOX[block[2]];  uint8_t s11 = AES_SBOX[block[7]];
    uint8_t s12 = AES_SBOX[block[12]]; uint8_t s13 = AES_SBOX[block[1]];
    uint8_t s14 = AES_SBOX[block[6]];  uint8_t s15 = AES_SBOX[block[11]];

    block[0]  = gf_mul2(s0)^gf_mul2(s1)^s1^s2^s3^key[0];
    block[1]  = s0^gf_mul2(s1)^gf_mul2(s2)^s2^s3^key[1];
    block[2]  = s0^s1^gf_mul2(s2)^gf_mul2(s3)^s3^key[2];
    block[3]  = gf_mul2(s0)^s0^s1^s2^gf_mul2(s3)^key[3];
    block[4]  = gf_mul2(s4)^gf_mul2(s5)^s5^s6^s7^key[4];
    block[5]  = s4^gf_mul2(s5)^gf_mul2(s6)^s6^s7^key[5];
    block[6]  = s4^s5^gf_mul2(s6)^gf_mul2(s7)^s7^key[6];
    block[7]  = gf_mul2(s4)^s4^s5^s6^gf_mul2(s7)^key[7];
    block[8]  = gf_mul2(s8)^gf_mul2(s9)^s9^s10^s11^key[8];
    block[9]  = s8^gf_mul2(s9)^gf_mul2(s10)^s10^s11^key[9];
    block[10] = s8^s9^gf_mul2(s10)^gf_mul2(s11)^s11^key[10];
    block[11] = gf_mul2(s8)^s8^s9^s10^gf_mul2(s11)^key[11];
    block[12] = gf_mul2(s12)^gf_mul2(s13)^s13^s14^s15^key[12];
    block[13] = s12^gf_mul2(s13)^gf_mul2(s14)^s14^s15^key[13];
    block[14] = s12^s13^gf_mul2(s14)^gf_mul2(s15)^s15^key[14];
    block[15] = gf_mul2(s12)^s12^s13^s14^gf_mul2(s15)^key[15];
}

// ===========================================================================
// STAGE 3 BRANCH OPERATIONS
// ===========================================================================

__device__ __noinline__ uint64_t stage3_isqrt_heavy(
    uint32_t sel, uint64_t a, uint64_t b, uint64_t c,
    uint64_t result, uint32_t i, uint32_t j)
{
    switch (sel) {
        case 0: {
            __uint128_t t = combine_u64(a + i, isqrt_v3(b + j));
            uint64_t denom = murmurhash3(c ^ result ^ i ^ j) | 1;
            return (uint64_t)(t % denom);
        }
        case 1: {
            uint64_t sq = isqrt_v3(b | 2);
            uint64_t t = (c + i) % sq;
            return rotl64(t, (i + j)) * isqrt_v3(a + j);
        }
        case 2:
            return (isqrt_v3(a + i) * isqrt_v3(c + j)) ^ (b + i + j);
        default:
            return 0;
    }
}

__device__ __noinline__ uint64_t stage3_heavy128(
    uint32_t sel, uint64_t a, uint64_t b, uint64_t c,
    uint64_t result, uint32_t r)
{
    if (sel >= 14) {
        if (sel == 14) return fast_mulhi_128_64(b, a, c);
        if (sel == 15) return fast_mulhi_256_128(a, c, rotr64(result, r), b);
        return 0;
    }
    switch (sel) {
        case 10: {
            __uint128_t n = combine_u64(a, b);
            return (uint64_t)(n % (c | 1));
        }
        case 11: {
            __uint128_t n = combine_u64(b, c);
            __uint128_t d = combine_u64(rotl64(result, r), a | 2);
            if (d > n) return c;
            return (uint64_t)(n % d);
        }
        case 12: {
            __uint128_t n = combine_u64(c, a);
            return (uint64_t)(n / (b | 4));
        }
        case 13: {
            __uint128_t n = combine_u64(rotl64(result, r), b);
            __uint128_t d = combine_u64(a, c | 8);
            if (n > d) return (uint64_t)(n / d);
            return a ^ b;
        }
        default: return 0;
    }
}

// ===========================================================================
// V4 COMPUTE BARRIER — Salsa20/8 + serial multiply chain
// ===========================================================================

__device__ __forceinline__ void salsa_qr(uint32_t s[16], int a, int b, int c, int d) {
    s[b] ^= __vadd4(s[a], s[d]); // just use wrapping add — rotates below
    // Proper Salsa20 QR:
    s[b] = 0; // reset — let me do this properly:
}

// Proper Salsa20 quarter-round
__device__ __forceinline__ void qr(uint32_t *s, int a, int b, int c, int d) {
    s[b] ^= __funnelshift_l(s[a] + s[d], s[a] + s[d], 7);
    s[c] ^= __funnelshift_l(s[b] + s[a], s[b] + s[a], 9);
    s[d] ^= __funnelshift_l(s[c] + s[b], s[c] + s[b], 13);
    s[a] ^= __funnelshift_l(s[d] + s[c], s[d] + s[c], 18);
}

__device__ __forceinline__ uint64_t compute_barrier(
    uint64_t v, uint64_t result, uint64_t a, uint64_t b, uint64_t c)
{
    // 1. Expand to 16 x u32 Salsa state
    uint32_t state[16];
    state[0]  = (uint32_t)v;           state[1]  = (uint32_t)(v >> 32);
    state[2]  = (uint32_t)result;      state[3]  = (uint32_t)(result >> 32);
    state[4]  = (uint32_t)a;           state[5]  = (uint32_t)(a >> 32);
    state[6]  = (uint32_t)b;           state[7]  = (uint32_t)(b >> 32);
    state[8]  = (uint32_t)c;           state[9]  = (uint32_t)(c >> 32);
    state[10] = (uint32_t)(v ^ a);     state[11] = (uint32_t)((v ^ a) >> 32);
    state[12] = (uint32_t)(result ^ b); state[13] = (uint32_t)((result ^ b) >> 32);
    state[14] = (uint32_t)(c ^ ~result); state[15] = (uint32_t)((c ^ ~result) >> 32);

    uint32_t orig[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) orig[k] = state[k];

    // 2. Salsa20/8: 4 double-rounds
    #pragma unroll
    for (int dr = 0; dr < 4; dr++) {
        // Column round
        qr(state, 0, 4, 8, 12);
        qr(state, 5, 9, 13, 1);
        qr(state, 10, 14, 2, 6);
        qr(state, 15, 3, 7, 11);
        // Row round
        qr(state, 0, 1, 2, 3);
        qr(state, 5, 6, 7, 4);
        qr(state, 10, 11, 8, 9);
        qr(state, 15, 12, 13, 14);
    }

    // 3. Feedforward
    #pragma unroll
    for (int k = 0; k < 16; k++) state[k] += orig[k];

    // 4. Serial multiply chain (strictly serial — latency floor)
    uint64_t m  = (uint64_t)state[0] | ((uint64_t)state[1] << 32);
    uint64_t m1 = (uint64_t)state[2] | ((uint64_t)state[3] << 32);
    uint64_t m2 = (uint64_t)state[4] | ((uint64_t)state[5] << 32);
    uint64_t m3 = (uint64_t)state[6] | ((uint64_t)state[7] << 32);

    m = ((uint64_t)(uint32_t)m) * (m >> 32) + m1;
    m = ((uint64_t)(uint32_t)m) * (m >> 32) + m2;
    m = ((uint64_t)(uint32_t)m) * (m >> 32) + m3;
    m = ((uint64_t)(uint32_t)m) * (m >> 32) + (m1 ^ m3);

    // 5. Fold tail
    uint64_t tail = ((uint64_t)state[8]  | ((uint64_t)state[9]  << 32))
                  ^ ((uint64_t)state[10] | ((uint64_t)state[11] << 32))
                  ^ ((uint64_t)state[12] | ((uint64_t)state[13] << 32))
                  ^ ((uint64_t)state[14] | ((uint64_t)state[15] << 32));

    return m ^ tail;
}

// ===========================================================================
// STAGE 3 KERNEL (parameterized: V3 or V4)
// ===========================================================================

template <bool USE_V4_BARRIER>
__global__ void xelis_stage3_kernel(uint64_t *all_scratch, uint32_t batch_size)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Each thread gets its own 544 KB scratchpad
    uint64_t *mem_buffer_a = all_scratch + (uint64_t)tid * MEMORY_SIZE;
    uint64_t *mem_buffer_b = mem_buffer_a + BUFFER_SIZE;

    const uint8_t key[16] = {'x','e','l','i','s','h','a','s','h','-','p','o','w','-','v',
                              USE_V4_BARRIER ? '4' : '3'};

    uint64_t addr_a = mem_buffer_b[BUFFER_SIZE - 1];
    uint64_t addr_b = mem_buffer_a[BUFFER_SIZE - 1] >> 32;
    uint32_t r = 0;

    for (uint32_t i = 0; i < SCRATCHPAD_ITERS; i++) {
        uint64_t idx_a_init = map_index(addr_a);
        uint64_t mem_a = mem_buffer_a[idx_a_init];
        uint64_t idx_b_init = map_index(mem_a ^ addr_b);
        uint64_t mem_b = mem_buffer_b[idx_b_init];

        uint8_t block[16];
        memcpy(block, &mem_b, 8);
        memcpy(block + 8, &mem_a, 8);
        aes_single_round(block, key);

        uint64_t hash1;
        memcpy(&hash1, block, 8);
        uint64_t hash2;
        memcpy(&hash2, block + 8, 8);
        uint64_t result = ~(hash1 ^ hash2);

        for (uint32_t j = 0; j < BUFFER_SIZE; j++) {
            uint64_t a = mem_buffer_a[map_index(result)];
            uint64_t b = mem_buffer_b[map_index(a ^ ~rotr64(result, r))];

            uint64_t c;
            if (r < BUFFER_SIZE)
                c = mem_buffer_a[r];
            else
                c = mem_buffer_b[r - BUFFER_SIZE];
            r = (r < MEMORY_SIZE - 1) ? r + 1 : 0;

            uint32_t sel = rotl64(result, c) & 0xf;

            uint64_t v;
            if (sel >= 10)
                v = stage3_heavy128(sel, a, b, c, result, r);
            else if (sel <= 2)
                v = stage3_isqrt_heavy(sel, a, b, c, result, i, j);
            else {
                switch (sel) {
                    case 3: v = (a + b) * c; break;
                    case 4: v = (b - c) * a; break;
                    case 5: v = c - a + b; break;
                    case 6: v = a - b + c; break;
                    case 7: v = b * c + a; break;
                    case 8: v = c * a + b; break;
                    case 9: v = a * b * c; break;
                    default: v = 0; break;
                }
            }

            // ===== V4 COMPUTE BARRIER (only difference) =====
            uint64_t effective_v;
            if (USE_V4_BARRIER) {
                effective_v = compute_barrier(v, result, a, b, c);
            } else {
                effective_v = v;
            }

            uint64_t idx_seed = effective_v ^ result;
            result = rotl64(idx_seed, r);

            bool use_buffer_b = pick_half(effective_v);
            uint64_t idx_t = map_index(idx_seed);
            uint64_t t_mem = use_buffer_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t];
            uint64_t t = t_mem ^ result;

            uint64_t w_idx_a = map_index(t ^ result ^ 0x9e3779b97f4a7c15ULL);
            uint64_t w_idx_b = map_index(w_idx_a ^ ~result ^ 0xd2b74407b1ce6e93ULL);

            uint64_t old_a = mem_buffer_a[w_idx_a];
            uint64_t old_b = mem_buffer_b[w_idx_b];

            mem_buffer_b[w_idx_b] = old_b ^ (old_a ^ rotr64(t, i + j));
            mem_buffer_a[w_idx_a] = t;
        }

        addr_a = modular_power(addr_a, addr_b, result);
        addr_b = isqrt_v3(result) * ((uint64_t)r + 1) * isqrt_v3(addr_a);
    }

    // Write back one word to prevent dead-code elimination
    mem_buffer_a[0] = addr_a ^ addr_b;
}

// ===========================================================================
// SIMPLE PRNG FILL KERNEL (xorshift64*)
// ===========================================================================

__global__ void fill_random(uint64_t *data, uint64_t num_elements, uint64_t seed_base)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // xorshift64* with per-element seed
    uint64_t s = seed_base + idx * 6364136223846793005ULL + 1442695040888963407ULL;
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    data[idx] = s * 0x2545F4914F6CDD1DULL;
}

// ===========================================================================
// HOST CODE
// ===========================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char **argv)
{
    int device_id = (argc > 1) ? atoi(argv[1]) : 0;
    int batch     = (argc > 2) ? atoi(argv[2]) : 0; // 0 = auto
    int runs      = (argc > 3) ? atoi(argv[3]) : 5;

    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("XelisHash V3 vs V4 — GPU Stage 3 Benchmark\n");
    printf("============================================\n");
    printf("Device %d: %s\n", device_id, prop.name);
    printf("SMs: %d, Mem: %.1f GB\n",
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Auto batch: use ~50% of VRAM, leave room for backup copy
    if (batch == 0) {
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        // Need 2x (original + backup) of batch * MEMORY_SIZE_BYTES
        size_t per_hash = (size_t)MEMORY_SIZE_BYTES;
        batch = (int)((free_mem * 0.4) / (per_hash * 2));
        // Round down to multiple of 128
        batch = (batch / 128) * 128;
        if (batch < 128) batch = 128;
    }

    size_t scratch_bytes = (size_t)batch * MEMORY_SIZE_BYTES;
    printf("Batch: %d hashes (%.1f MB scratchpad)\n", batch, scratch_bytes / (1024.0 * 1024.0));
    printf("Runs: %d\n", runs);
    printf("Inner iterations: %u x %u = %u per hash\n\n",
           SCRATCHPAD_ITERS, BUFFER_SIZE, SCRATCHPAD_ITERS * BUFFER_SIZE);

    // Allocate
    uint64_t *d_scratch, *d_backup;
    CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
    CUDA_CHECK(cudaMalloc(&d_backup, scratch_bytes));

    // Fill with random data
    printf("Filling %d scratchpads with random data...\n", batch);
    {
        uint64_t num_elements = (uint64_t)batch * MEMORY_SIZE;
        int threads = 256;
        int blocks = (int)((num_elements + threads - 1) / threads);
        fill_random<<<blocks, threads>>>(d_scratch, num_elements, 0xDEADBEEF42ULL);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Backup the random scratchpads
    CUDA_CHECK(cudaMemcpy(d_backup, d_scratch, scratch_bytes, cudaMemcpyDeviceToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int tpb = 128;
    int grid = (batch + tpb - 1) / tpb;

    // ── Warmup ──────────────────────────────────────────────────────────────
    printf("Warming up...\n");
    xelis_stage3_kernel<false><<<grid, tpb>>>(d_scratch, batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_scratch, d_backup, scratch_bytes, cudaMemcpyDeviceToDevice));

    xelis_stage3_kernel<true><<<grid, tpb>>>(d_scratch, batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_scratch, d_backup, scratch_bytes, cudaMemcpyDeviceToDevice));

    // ── Benchmark V3 ────────────────────────────────────────────────────────
    printf("\nBenchmarking V3 (original)...\n");
    float v3_times[32];
    for (int r_idx = 0; r_idx < runs; r_idx++) {
        // Restore random data before each run
        CUDA_CHECK(cudaMemcpy(d_scratch, d_backup, scratch_bytes, cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaEventRecord(start));
        xelis_stage3_kernel<false><<<grid, tpb>>>(d_scratch, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        v3_times[r_idx] = ms;
        float khs = batch / (ms / 1000.0f) / 1000.0f;
        printf("  Run %d: %.2f ms (%.2f kH/s)\n", r_idx + 1, ms, khs);
    }

    // ── Benchmark V4 ────────────────────────────────────────────────────────
    printf("\nBenchmarking V4 (with compute barrier)...\n");
    float v4_times[32];
    for (int r_idx = 0; r_idx < runs; r_idx++) {
        // Restore random data before each run
        CUDA_CHECK(cudaMemcpy(d_scratch, d_backup, scratch_bytes, cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaEventRecord(start));
        xelis_stage3_kernel<true><<<grid, tpb>>>(d_scratch, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        v4_times[r_idx] = ms;
        float khs = batch / (ms / 1000.0f) / 1000.0f;
        printf("  Run %d: %.2f ms (%.2f kH/s)\n", r_idx + 1, ms, khs);
    }

    // ── Results ─────────────────────────────────────────────────────────────
    // Compute averages (skip first run)
    int skip = (runs > 2) ? 1 : 0;
    float v3_avg = 0, v4_avg = 0;
    int count = runs - skip;
    for (int i = skip; i < runs; i++) {
        v3_avg += v3_times[i];
        v4_avg += v4_times[i];
    }
    v3_avg /= count;
    v4_avg /= count;

    float v3_khs = batch / (v3_avg / 1000.0f) / 1000.0f;
    float v4_khs = batch / (v4_avg / 1000.0f) / 1000.0f;
    float ratio = v4_avg / v3_avg;

    printf("\n============================================\n");
    printf("RESULTS (%s, device %d)\n", prop.name, device_id);
    printf("============================================\n");
    printf("V3 (original):        %.2f ms avg -> %.2f kH/s\n", v3_avg, v3_khs);
    printf("V4 (compute barrier): %.2f ms avg -> %.2f kH/s\n", v4_avg, v4_khs);
    printf("Ratio: V4 is %.2fx slower (%.1f%% overhead)\n", ratio, (ratio - 1.0f) * 100.0f);
    printf("V4/V3 throughput: %.1f%%\n", 100.0f / ratio);
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_backup));

    return 0;
}
