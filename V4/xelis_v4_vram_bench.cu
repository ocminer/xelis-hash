// xelis_v4_vram_bench.cu — V3 vs V4-VRAM (large per-thread V-array)
//
// Strategy: Replace 2KB S-box with 2MB per-thread V-array in global memory.
// VRAM capacity limits occupancy naturally — no bypass possible.
// CPU: V-array fits in L3 (~15-30 cycle access).  GPU: VRAM (~400 cycle access).
// L2 can't cache it: 2MB × thousands of threads >> 98MB L2.
//
// Build: nvcc -O3 -arch=sm_120 -o bench_vram xelis_v4_vram_bench.cu --use_fast_math
// Run:   ./bench_vram [device] [batch] [runs]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define MEMORY_SIZE      (531u * 128u)    // 67,968 u64s = 544 KB (V3 scratchpad)
#define BUFFER_SIZE      (MEMORY_SIZE / 2u)
#define SCRATCHPAD_ITERS 2u

// V4-VRAM parameters
#ifndef VARRAY_SIZE
#define VARRAY_SIZE      (256u * 1024u)   // 262,144 u64s = 2 MB per thread
#endif
#ifndef VARRAY_READS
#define VARRAY_READS     2                // dependent reads per inner iteration
#endif
#define STATE_WORDS      8

// ===========================================================================
// DEVICE HELPERS
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

__device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) {
    n &= 63u; return (x << n) | (x >> ((64u - n) & 63u));
}
__device__ __forceinline__ uint64_t rotr64(uint64_t x, uint32_t n) {
    n &= 63u; return (x >> n) | (x << ((64u - n) & 63u));
}
__device__ __forceinline__ uint64_t murmurhash3(uint64_t s) {
    s ^= s >> 55; s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 32; s *= 0xc4ceb9fe1a85ec53ULL;
    s ^= s >> 15; return s;
}
__device__ __forceinline__ uint64_t map_index(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    uint32_t xhi = (uint32_t)(x >> 32), xlo = (uint32_t)x;
    uint32_t carry = (uint32_t)(((uint64_t)xlo * BUFFER_SIZE) >> 32);
    return ((uint64_t)xhi * BUFFER_SIZE + carry) >> 32;
}
// Map to V-array range [0, VARRAY_SIZE)
__device__ __forceinline__ uint64_t map_varray(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    uint32_t xhi = (uint32_t)(x >> 32), xlo = (uint32_t)x;
    uint32_t carry = (uint32_t)(((uint64_t)xlo * VARRAY_SIZE) >> 32);
    return ((uint64_t)xhi * VARRAY_SIZE + carry) >> 32;
}
__device__ __forceinline__ bool pick_half(uint64_t s) {
    return (murmurhash3(s) & (1ULL << 58)) != 0;
}
__device__ __forceinline__ uint64_t isqrt_v3(uint64_t n) {
    if (n < 2) return n;
    uint64_t x = __double2ull_rz(__dsqrt_rn(__ull2double_rn(n)));
    if (x > 4294967295ULL) x = 4294967295ULL;
    uint64_t sq = x * x;
    if (sq > n || (x > 0 && sq / x != x)) { x--; sq = x*x; if (sq > n) x--; }
    else { uint64_t g=n-sq, nd=2*x+1; if(g>=nd){x++;sq=x*x;g=n-sq;nd=2*x+1;if(g>=nd)x++;} }
    return x;
}
__device__ __forceinline__ uint64_t modular_power(uint64_t base, uint64_t exp, uint64_t mod) {
    if (mod == 0) return 0;
    uint64_t r = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) r = (uint64_t)((__uint128_t)r * base % mod);
        base = (uint64_t)((__uint128_t)base * base % mod);
        exp >>= 1;
    }
    return r;
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
__device__ __forceinline__ uint8_t gf_mul2(uint8_t x) {
    return (x << 1) ^ ((x >> 7) * 0x1b);
}
__device__ __forceinline__ void aes_single_round(uint8_t block[16], const uint8_t key[16]) {
    uint8_t s0=AES_SBOX[block[0]],s1=AES_SBOX[block[5]],s2=AES_SBOX[block[10]],s3=AES_SBOX[block[15]];
    uint8_t s4=AES_SBOX[block[4]],s5=AES_SBOX[block[9]],s6=AES_SBOX[block[14]],s7=AES_SBOX[block[3]];
    uint8_t s8=AES_SBOX[block[8]],s9=AES_SBOX[block[13]],s10=AES_SBOX[block[2]],s11=AES_SBOX[block[7]];
    uint8_t s12=AES_SBOX[block[12]],s13=AES_SBOX[block[1]],s14=AES_SBOX[block[6]],s15=AES_SBOX[block[11]];
    block[0]=gf_mul2(s0)^gf_mul2(s1)^s1^s2^s3^key[0]; block[1]=s0^gf_mul2(s1)^gf_mul2(s2)^s2^s3^key[1];
    block[2]=s0^s1^gf_mul2(s2)^gf_mul2(s3)^s3^key[2]; block[3]=gf_mul2(s0)^s0^s1^s2^gf_mul2(s3)^key[3];
    block[4]=gf_mul2(s4)^gf_mul2(s5)^s5^s6^s7^key[4]; block[5]=s4^gf_mul2(s5)^gf_mul2(s6)^s6^s7^key[5];
    block[6]=s4^s5^gf_mul2(s6)^gf_mul2(s7)^s7^key[6]; block[7]=gf_mul2(s4)^s4^s5^s6^gf_mul2(s7)^key[7];
    block[8]=gf_mul2(s8)^gf_mul2(s9)^s9^s10^s11^key[8]; block[9]=s8^gf_mul2(s9)^gf_mul2(s10)^s10^s11^key[9];
    block[10]=s8^s9^gf_mul2(s10)^gf_mul2(s11)^s11^key[10]; block[11]=gf_mul2(s8)^s8^s9^s10^gf_mul2(s11)^key[11];
    block[12]=gf_mul2(s12)^gf_mul2(s13)^s13^s14^s15^key[12]; block[13]=s12^gf_mul2(s13)^gf_mul2(s14)^s14^s15^key[13];
    block[14]=s12^s13^gf_mul2(s14)^gf_mul2(s15)^s15^key[14]; block[15]=gf_mul2(s12)^s12^s13^s14^gf_mul2(s15)^key[15];
}

__device__ __noinline__ uint64_t stage3_isqrt_heavy(uint32_t sel, uint64_t a, uint64_t b, uint64_t c, uint64_t result, uint32_t i, uint32_t j) {
    switch (sel) {
        case 0: { __uint128_t t=combine_u64(a+i,isqrt_v3(b+j)); return (uint64_t)(t%(murmurhash3(c^result^i^j)|1)); }
        case 1: { uint64_t sq=isqrt_v3(b|2); return rotl64((c+i)%sq,(i+j))*isqrt_v3(a+j); }
        case 2: return (isqrt_v3(a+i)*isqrt_v3(c+j))^(b+i+j);
        default: return 0;
    }
}
__device__ __noinline__ uint64_t stage3_heavy128(uint32_t sel, uint64_t a, uint64_t b, uint64_t c, uint64_t result, uint32_t r) {
    if (sel==14) return fast_mulhi_128_64(b,a,c);
    if (sel==15) return fast_mulhi_256_128(a,c,rotr64(result,r),b);
    switch(sel) {
        case 10: return (uint64_t)(combine_u64(a,b)%(c|1));
        case 11: { __uint128_t n=combine_u64(b,c),d=combine_u64(rotl64(result,r),a|2); return d>n?c:(uint64_t)(n%d); }
        case 12: return (uint64_t)(combine_u64(c,a)/(b|4));
        case 13: { __uint128_t n=combine_u64(rotl64(result,r),b),d=combine_u64(a,c|8); return n>d?(uint64_t)(n/d):a^b; }
        default: return 0;
    }
}

// ===========================================================================
// V3 KERNEL (baseline)
// ===========================================================================
__global__ void xelis_stage3_v3(uint64_t *all_scratch, uint32_t batch_size)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    uint64_t *mem_buffer_a = all_scratch + (uint64_t)tid * MEMORY_SIZE;
    uint64_t *mem_buffer_b = mem_buffer_a + BUFFER_SIZE;
    const uint8_t key[16] = {'x','e','l','i','s','h','a','s','h','-','p','o','w','-','v','3'};
    uint64_t addr_a = mem_buffer_b[BUFFER_SIZE-1];
    uint64_t addr_b = mem_buffer_a[BUFFER_SIZE-1] >> 32;
    uint32_t r = 0;
    for (uint32_t i = 0; i < SCRATCHPAD_ITERS; i++) {
        uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
        uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];
        uint8_t block[16]; memcpy(block,&mem_b,8); memcpy(block+8,&mem_a,8);
        aes_single_round(block, key);
        uint64_t h1; memcpy(&h1,block,8);
        uint64_t h2; memcpy(&h2,block+8,8);
        uint64_t result = ~(h1^h2);
        for (uint32_t j = 0; j < BUFFER_SIZE; j++) {
            uint64_t a = mem_buffer_a[map_index(result)];
            uint64_t b = mem_buffer_b[map_index(a ^ ~rotr64(result,r))];
            uint64_t c = (r<BUFFER_SIZE) ? mem_buffer_a[r] : mem_buffer_b[r-BUFFER_SIZE];
            r = (r<MEMORY_SIZE-1) ? r+1 : 0;
            uint32_t sel = rotl64(result,c) & 0xf;
            uint64_t v;
            if (sel>=10) v=stage3_heavy128(sel,a,b,c,result,r);
            else if (sel<=2) v=stage3_isqrt_heavy(sel,a,b,c,result,i,j);
            else switch(sel) {
                case 3:v=(a+b)*c;break; case 4:v=(b-c)*a;break; case 5:v=c-a+b;break;
                case 6:v=a-b+c;break; case 7:v=b*c+a;break; case 8:v=c*a+b;break;
                case 9:v=a*b*c;break; default:v=0;
            }
            uint64_t idx_seed = v ^ result;
            result = rotl64(idx_seed, r);
            bool use_b = pick_half(v);
            uint64_t idx_t = map_index(idx_seed);
            uint64_t t = (use_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;
            uint64_t w_a = map_index(t^result^0x9e3779b97f4a7c15ULL);
            uint64_t w_b = map_index(w_a^~result^0xd2b74407b1ce6e93ULL);
            uint64_t old_a = mem_buffer_a[w_a], old_b = mem_buffer_b[w_b];
            mem_buffer_b[w_b] = old_b ^ (old_a ^ rotr64(t, i+j));
            mem_buffer_a[w_a] = t;
        }
        addr_a = modular_power(addr_a, addr_b, result);
        addr_b = isqrt_v3(result) * ((uint64_t)r+1) * isqrt_v3(addr_a);
    }
    mem_buffer_a[0] = addr_a ^ addr_b;
}

// ===========================================================================
// V4-VRAM KERNEL — per-thread V-array in global memory
// ===========================================================================
// No shared memory needed. VRAM capacity naturally limits occupancy.
// 2MB per thread: RTX 5090 (32GB) → ~12K threads → ~2.2 warps/SM
//                 CMP 170HX (8GB) → ~3K threads → ~1.3 warps/SM
// L2 overflow: 12K threads × 2MB = 24GB >> 98MB L2 → every access hits VRAM.

__global__ void xelis_stage3_v4_vram(uint64_t *all_scratch, uint64_t *all_varray, uint32_t batch_size)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    uint64_t *mem_buffer_a = all_scratch + (uint64_t)tid * MEMORY_SIZE;
    uint64_t *mem_buffer_b = mem_buffer_a + BUFFER_SIZE;
    uint64_t *varray = all_varray + (uint64_t)tid * VARRAY_SIZE;

    // ---- Fill V-array sequentially (Argon2d-style) ----
    // Each entry depends on previous entry + scratchpad data.
    // Sequential dependency prevents parallel fill tricks.
    {
        uint64_t prev = mem_buffer_a[0] ^ 0x6A09E667F3BCC908ULL;
        for (uint32_t k = 0; k < VARRAY_SIZE; k++) {
            // Mix previous with scratchpad (wrapping) — serial dependency
            uint64_t sp_val = mem_buffer_a[k % MEMORY_SIZE];
            prev = prev ^ sp_val;
            prev = murmurhash3(prev);
            varray[k] = prev;
        }
    }

    // ---- Init state ----
    uint64_t state[STATE_WORDS];
    #pragma unroll
    for (int k = 0; k < STATE_WORDS; k++)
        state[k] = varray[k];

    const uint8_t key[16] = {'x','e','l','i','s','h','a','s','h','-','p','o','w','-','v','4'};
    uint64_t addr_a = mem_buffer_b[BUFFER_SIZE-1];
    uint64_t addr_b = mem_buffer_a[BUFFER_SIZE-1] >> 32;
    uint32_t r = 0;

    for (uint32_t i = 0; i < SCRATCHPAD_ITERS; i++) {
        uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
        uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];
        uint8_t block[16]; memcpy(block,&mem_b,8); memcpy(block+8,&mem_a,8);
        aes_single_round(block, key);
        uint64_t h1; memcpy(&h1,block,8);
        uint64_t h2; memcpy(&h2,block+8,8);
        uint64_t result = ~(h1^h2);

        for (uint32_t j = 0; j < BUFFER_SIZE; j++) {
            // ====== FULL V3 MEMORY ACCESS PATTERN (unchanged) ======
            uint64_t a = mem_buffer_a[map_index(result)];
            uint64_t b = mem_buffer_b[map_index(a ^ ~rotr64(result,r))];
            uint64_t c = (r<BUFFER_SIZE) ? mem_buffer_a[r] : mem_buffer_b[r-BUFFER_SIZE];
            r = (r<MEMORY_SIZE-1) ? r+1 : 0;

            uint32_t sel = rotl64(result,c) & 0xf;
            uint64_t v;
            if (sel>=10) v=stage3_heavy128(sel,a,b,c,result,r);
            else if (sel<=2) v=stage3_isqrt_heavy(sel,a,b,c,result,i,j);
            else switch(sel) {
                case 3:v=(a+b)*c;break; case 4:v=(b-c)*a;break; case 5:v=c-a+b;break;
                case 6:v=a-b+c;break; case 7:v=b*c+a;break; case 8:v=c*a+b;break;
                case 9:v=a*b*c;break; default:v=0;
            }

            // ====== V4-VRAM: Dependent V-array reads ======
            // Each read index depends on previous read result → serial through VRAM.
            // On CPU: these hit L3 (~15-30 cycles). On GPU: VRAM (~400 cycles each).
            uint64_t chain = v;
            #pragma unroll
            for (int m = 0; m < VARRAY_READS; m++) {
                uint64_t vidx = map_varray(chain ^ result ^ ((uint64_t)m * 0x9E3779B97F4A7C15ULL));
                chain ^= varray[vidx];
                chain = rotl64(chain, 17);
            }

            // State mixing
            uint64_t si = state[(result >> 4) & (STATE_WORDS-1)];
            chain ^= si;
            state[(result >> 10) & (STATE_WORDS-1)] = chain;

            // V-array write-back (evolves the array, prevents caching tricks)
            uint64_t widx = map_varray(chain ^ 0xD2B74407B1CE6E93ULL);
            varray[widx] ^= chain;

            uint64_t idx_seed = (v ^ chain) ^ result;
            // ====== END V4-VRAM ======

            result = rotl64(idx_seed, r);
            bool use_b = pick_half(v);
            uint64_t idx_t = map_index(idx_seed);
            uint64_t t = (use_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;
            uint64_t w_a = map_index(t^result^0x9e3779b97f4a7c15ULL);
            uint64_t w_b = map_index(w_a^~result^0xd2b74407b1ce6e93ULL);
            uint64_t old_a = mem_buffer_a[w_a], old_b = mem_buffer_b[w_b];
            mem_buffer_b[w_b] = old_b ^ (old_a ^ rotr64(t, i+j));
            mem_buffer_a[w_a] = t;
        }

        addr_a = modular_power(addr_a, addr_b, result);
        addr_b = isqrt_v3(result) * ((uint64_t)r+1) * isqrt_v3(addr_a);
    }
    mem_buffer_a[0] = addr_a ^ addr_b ^ state[0];
}

// ===========================================================================
// RANDOM FILL + HOST
// ===========================================================================
__global__ void fill_random(uint64_t *data, uint64_t num, uint64_t seed) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    uint64_t s = seed + idx * 6364136223846793005ULL + 1442695040888963407ULL;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    data[idx] = s * 0x2545F4914F6CDD1DULL;
}

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

int main(int argc, char **argv) {
    int dev   = argc > 1 ? atoi(argv[1]) : 0;
    int batch = argc > 2 ? atoi(argv[2]) : 0;
    int runs  = argc > 3 ? atoi(argv[3]) : 5;

    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("XelisHash V3 vs V4-VRAM Benchmark\n");
    printf("==================================\n");
    printf("Device %d: %s (%d SMs, %.1f GB)\n", dev, prop.name,
           prop.multiProcessorCount, prop.totalGlobalMem/(1024.0*1024.0*1024.0));
    printf("V-array: %u entries = %.1f MB per thread\n", VARRAY_SIZE, (double)VARRAY_SIZE*8/1024/1024);
    printf("V-array reads per iter: %d\n", VARRAY_READS);

    if (batch == 0) {
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        // Per hash: scratchpad (MEMORY_SIZE*8) + backup + V-array (VARRAY_SIZE*8)
        size_t per_hash = (size_t)MEMORY_SIZE * 8 * 2 + (size_t)VARRAY_SIZE * 8;
        batch = (int)((free_mem * 0.85) / per_hash);
        batch = (batch / 128) * 128;  // align
        if (batch < 128) batch = 128;
    }

    size_t scratch_bytes = (size_t)batch * MEMORY_SIZE * 8;
    size_t varray_bytes = (size_t)batch * VARRAY_SIZE * 8;
    printf("Batch: %d hashes\n", batch);
    printf("Scratchpad: %.1f MB, V-array: %.1f MB, Total: %.1f GB\n",
           scratch_bytes/1024.0/1024.0, varray_bytes/1024.0/1024.0,
           (scratch_bytes*2+varray_bytes)/1024.0/1024.0/1024.0);
    printf("Threads/SM: %.1f, Warps/SM: %.1f\n",
           (float)batch/prop.multiProcessorCount,
           (float)batch/prop.multiProcessorCount/32.0);
    printf("Runs: %d\n\n", runs);

    uint64_t *d_scratch, *d_backup, *d_varray;
    CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
    CUDA_CHECK(cudaMalloc(&d_backup, scratch_bytes));
    CUDA_CHECK(cudaMalloc(&d_varray, varray_bytes));

    // Fill random
    { uint64_t n=(uint64_t)batch*MEMORY_SIZE; int t=256,b2=(int)((n+t-1)/t);
      fill_random<<<b2,t>>>(d_scratch,n,0xCAFEBABE42ULL);
      CUDA_CHECK(cudaDeviceSynchronize()); }
    CUDA_CHECK(cudaMemcpy(d_backup, d_scratch, scratch_bytes, cudaMemcpyDeviceToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int tpb = 128;
    int grid = (batch + tpb - 1) / tpb;

    // Warmup
    CUDA_CHECK(cudaMemcpy(d_scratch, d_backup, scratch_bytes, cudaMemcpyDeviceToDevice));
    xelis_stage3_v3<<<grid, tpb>>>(d_scratch, batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_scratch, d_backup, scratch_bytes, cudaMemcpyDeviceToDevice));
    xelis_stage3_v4_vram<<<grid, tpb>>>(d_scratch, d_varray, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark V3
    printf("V3 (baseline):\n");
    float v3_avg = 0;
    for (int r_idx = 0; r_idx < runs; r_idx++) {
        CUDA_CHECK(cudaMemcpy(d_scratch, d_backup, scratch_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        xelis_stage3_v3<<<grid, tpb>>>(d_scratch, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float khs = batch / (ms / 1000.0f) / 1000.0f;
        printf("  Run %d: %.2f ms (%.2f kH/s)\n", r_idx+1, ms, khs);
        if (r_idx > 0) v3_avg += ms;
    }
    v3_avg /= (runs > 1 ? runs-1 : 1);

    // Benchmark V4-VRAM
    printf("\nV4-VRAM (%d reads, %.1f MB/thread):\n", VARRAY_READS, (double)VARRAY_SIZE*8/1024/1024);
    float v4_avg = 0;
    for (int r_idx = 0; r_idx < runs; r_idx++) {
        CUDA_CHECK(cudaMemcpy(d_scratch, d_backup, scratch_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        xelis_stage3_v4_vram<<<grid, tpb>>>(d_scratch, d_varray, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float khs = batch / (ms / 1000.0f) / 1000.0f;
        printf("  Run %d: %.2f ms (%.2f kH/s)\n", r_idx+1, ms, khs);
        if (r_idx > 0) v4_avg += ms;
    }
    v4_avg /= (runs > 1 ? runs-1 : 1);

    float v3_khs = batch / (v3_avg/1000.0f) / 1000.0f;
    float v4_khs = batch / (v4_avg/1000.0f) / 1000.0f;

    printf("\n==================================\n");
    printf("RESULTS: %s (device %d, %d SMs)\n", prop.name, dev, prop.multiProcessorCount);
    printf("==================================\n");
    printf("V3:      %.2f ms -> %.2f kH/s (baseline)\n", v3_avg, v3_khs);
    printf("V4-VRAM: %.2f ms -> %.2f kH/s (%d reads, %.1f MB/thread)\n",
           v4_avg, v4_khs, VARRAY_READS, (double)VARRAY_SIZE*8/1024/1024);
    printf("Ratio: V4-VRAM is %.2fx slower than V3\n", v4_avg / v3_avg);
    printf("Per-SM: V3=%.1f H/s  V4=%.1f H/s\n",
           v3_khs*1000.0f/prop.multiProcessorCount,
           v4_khs*1000.0f/prop.multiProcessorCount);
    printf("\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_varray));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_backup));
    return 0;
}
