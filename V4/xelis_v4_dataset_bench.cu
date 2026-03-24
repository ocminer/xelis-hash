// xelis_v4_dataset_bench.cu — S-box + TapeMix + Shared Dataset
//
// Three mechanisms:
// 1. S-box (2KB shared mem) → limits GPU to 2 warps/SM
// 2. TapeMix (per-hash tape) → warp divergence at low occupancy
// 3. Shared 256MB dataset → forces DRAM bandwidth competition on CPU
//    - All CPU cores share ONE dataset → ~50 GB/s DRAM bottleneck caps scaling
//    - GPU: dataset in VRAM, ~1800 GB/s → not the bottleneck
//    - On CPU, 32 cores × random 8B reads at ~70ns DRAM = bandwidth-limited
//
// Dataset is deterministic, generated once from a seed (like RandomX's dataset).
// Contains the SUPRNOVA marker: 0x5355_5052_4E4F_5641

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#ifndef TAPE_LEN
#define TAPE_LEN 32
#endif
#ifndef TAPE_FREQ
#define TAPE_FREQ 32
#endif
#ifndef DATASET_SIZE_MB
#define DATASET_SIZE_MB 256
#endif
#ifndef DATASET_READS
#define DATASET_READS 2  // random dataset reads per inner iteration
#endif

#define DATASET_SIZE     ((uint64_t)DATASET_SIZE_MB * 1024 * 1024 / 8)  // in u64s
#define MEMORY_SIZE      (531u * 128u)
#define BUFFER_SIZE      (MEMORY_SIZE / 2u)
#define SCRATCHPAD_ITERS 2u
#define SBOX_ENTRIES     256
#define STATE_WORDS      8

// SUPRNOVA marker: "SUPRNOVA" = 0x53 0x55 0x50 0x52 0x4E 0x4F 0x56 0x41
#define SUPRNOVA_MAGIC   0x5355505250524E4FULL  // "SUPRRPNO" little-endian mix
#define SUPRNOVA_SEED    0x5355505249564F41ULL  // "SUPRNOVA" as u64 (reversed bytes)

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

__device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) { n &= 63u; return (x << n) | (x >> ((64u - n) & 63u)); }
__device__ __forceinline__ uint64_t rotr64(uint64_t x, uint32_t n) { n &= 63u; return (x >> n) | (x << ((64u - n) & 63u)); }
__device__ __forceinline__ uint64_t murmurhash3(uint64_t s) {
    s ^= s >> 55; s *= 0xff51afd7ed558ccdULL; s ^= s >> 32; s *= 0xc4ceb9fe1a85ec53ULL; s ^= s >> 15; return s;
}
__device__ __forceinline__ uint64_t map_index(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    uint32_t xhi = (uint32_t)(x >> 32), xlo = (uint32_t)x;
    uint32_t carry = (uint32_t)(((uint64_t)xlo * BUFFER_SIZE) >> 32);
    return ((uint64_t)xhi * BUFFER_SIZE + carry) >> 32;
}
__device__ __forceinline__ uint64_t map_dataset(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    uint32_t xhi = (uint32_t)(x >> 32), xlo = (uint32_t)x;
    uint32_t carry = (uint32_t)(((uint64_t)xlo * DATASET_SIZE) >> 32);
    return ((uint64_t)xhi * DATASET_SIZE + carry) >> 32;
}
__device__ __forceinline__ bool pick_half(uint64_t s) { return (murmurhash3(s) & (1ULL << 58)) != 0; }
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
    if (mod == 0) return 0; uint64_t r = 1; base %= mod;
    while (exp > 0) { if (exp & 1) r = (uint64_t)((__uint128_t)r * base % mod); base = (uint64_t)((__uint128_t)base * base % mod); exp >>= 1; }
    return r;
}
__device__ __forceinline__ __uint128_t combine_u64(uint64_t hi, uint64_t lo) { return ((__uint128_t)hi << 64) | lo; }
__device__ __forceinline__ uint64_t fast_mulhi_128_64(uint64_t hi, uint64_t lo, uint64_t m) { return __umul64hi(lo, m) + hi * m; }
__device__ __forceinline__ uint64_t fast_mulhi_256_128(uint64_t a, uint64_t c, uint64_t d, uint64_t b) { return __umul64hi(c, b) + a * b + c * d; }
__device__ __forceinline__ uint8_t gf_mul2(uint8_t x) { return (x << 1) ^ ((x >> 7) * 0x1b); }
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

__device__ __noinline__ uint64_t tape_exec(uint32_t opcode, uint64_t s0, uint64_t s1, uint64_t operand) {
    switch (opcode & 0xF) {
        case 0:  return s0 ^ operand;
        case 1:  return s0 + operand;
        case 2:  return rotl64(s0, operand & 63);
        case 3:  return s0 * (operand | 1);
        case 4:  return s0 ^ rotl64(s1, 17) ^ operand;
        case 5:  return (s0 + s1) * (operand | 1);
        case 6:  return s0 ^ murmurhash3(operand);
        case 7:  return rotl64(s0 * s1, operand & 63) ^ operand;
        case 8:  return s0 ^ (s1 >> (operand & 31));
        case 9:  return (s0 & operand) | (s1 & ~operand);
        case 10: return murmurhash3(s0 ^ s1) + operand;
        case 11: return __umul64hi(s0, operand) ^ s1;
        case 12: return isqrt_v3(s0 ^ operand);
        case 13: {
            uint8_t blk[16], k[16];
            memcpy(blk, &s0, 8); memcpy(blk+8, &s1, 8);
            memcpy(k, &operand, 8); memcpy(k+8, &operand, 8);
            aes_single_round(blk, k);
            uint64_t r; memcpy(&r, blk, 8); return r;
        }
        case 14: return modular_power(s0 & 0xFFFF, s1 & 0x1F, operand | 3);
        case 15: return (uint64_t)(combine_u64(s0, s1) % (operand | 1));
        default: return s0;
    }
}

// Generate the shared dataset on GPU (deterministic from seed)
__global__ void generate_dataset(uint64_t *dataset, uint64_t size) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // Seed with SUPRNOVA marker
    uint64_t s = SUPRNOVA_SEED ^ (idx * 0x9E3779B97F4A7C15ULL);
    s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ULL;
    s ^= s >> 27; s *= 0x94D049BB133111EBULL;
    s ^= s >> 31;
    // Every 4096th entry gets the raw SUPRNOVA marker XOR'd in
    if ((idx & 0xFFF) == 0x53) s ^= SUPRNOVA_SEED;
    dataset[idx] = s;
}

// ========================================================================
// FULL V4 KERNEL: S-box + TapeMix + Dataset reads
// ========================================================================
__global__ void __launch_bounds__(32, 2)
xelis_stage3_full(uint64_t *all_scratch, const uint64_t *dataset, uint32_t batch_size)
{
    extern __shared__ uint64_t shared_sbox[];
    uint64_t *sbox = &shared_sbox[threadIdx.x * SBOX_ENTRIES];
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    uint64_t *mem_buffer_a = all_scratch + (uint64_t)tid * MEMORY_SIZE;
    uint64_t *mem_buffer_b = mem_buffer_a + BUFFER_SIZE;

    #pragma unroll 4
    for (int k = 0; k < SBOX_ENTRIES; k++)
        sbox[k] = mem_buffer_a[k] ^ (0xA5A5A5A5A5A5A5A5ULL + k);

    uint64_t state[STATE_WORDS];
    #pragma unroll
    for (int k = 0; k < STATE_WORDS; k++)
        state[k] = mem_buffer_a[SBOX_ENTRIES + k];

    uint8_t tape[TAPE_LEN];
    for (int k = 0; k < TAPE_LEN; k++)
        tape[k] = (uint8_t)(mem_buffer_a[k * (BUFFER_SIZE / TAPE_LEN)] >> 56) & 0xF;

    const uint8_t key[16] = {'x','e','l','i','s','h','a','s','h','-','p','o','w','-','v','4'};
    uint64_t addr_a = mem_buffer_b[BUFFER_SIZE-1];
    uint64_t addr_b = mem_buffer_a[BUFFER_SIZE-1] >> 32;
    uint32_t r = 0;
    uint64_t tape_state = addr_a ^ addr_b;

    for (uint32_t i = 0; i < SCRATCHPAD_ITERS; i++) {
        uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
        uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];
        uint8_t block[16]; memcpy(block,&mem_b,8); memcpy(block+8,&mem_a,8);
        aes_single_round(block, key);
        uint64_t h1; memcpy(&h1,block,8); uint64_t h2; memcpy(&h2,block+8,8);
        uint64_t result = ~(h1^h2);

        for (uint32_t j = 0; j < BUFFER_SIZE; j++) {
            uint64_t a = mem_buffer_a[map_index(result)];
            uint64_t b = mem_buffer_b[map_index(a ^ ~rotr64(result,r))];
            uint64_t c = (r<BUFFER_SIZE) ? mem_buffer_a[r] : mem_buffer_b[r-BUFFER_SIZE];
            r = (r<MEMORY_SIZE-1) ? r+1 : 0;

            // S-box
            uint64_t chain = a ^ sbox[(a >> 56) & 0xFF];
            state[((result >> 10) & (STATE_WORDS-1))] = chain;
            sbox[(chain >> 48) & 0xFF] ^= chain;

            // Dataset reads: random page every 64 iters, sequential within page.
            // CPU: prefetcher handles sequential within page (~5ns/read).
            //      Page hop every 64 iters → 1 cache miss per 64 iters (~0.5ns amortized).
            // GPU: 32 threads hop to 32 random pages → scattered VRAM access every 64 iters.
            //      Within page: per-thread sequential, but no cross-thread coalescing.
            #if DATASET_READS > 0
            {
                const uint32_t PAGE_SIZE = 512;  // 4KB page in u64s
                uint64_t page_idx;
                if ((j & 63) == 0) {
                    page_idx = map_dataset(result ^ chain);
                    page_idx = (page_idx / PAGE_SIZE) * PAGE_SIZE;
                } else {
                    page_idx = ((addr_a + j) % DATASET_SIZE);
                    page_idx = (page_idx / PAGE_SIZE) * PAGE_SIZE;
                }
                uint32_t offset = j & (PAGE_SIZE - 1);
                #pragma unroll
                for (int dr = 0; dr < DATASET_READS; dr++) {
                    chain ^= dataset[page_idx + ((offset + dr) % PAGE_SIZE)];
                }
                chain = rotl64(chain, 13);
            }
            #endif

            // TapeMix
            if ((j & (TAPE_FREQ - 1)) == 0) {
                #pragma unroll 1
                for (int t = 0; t < TAPE_LEN; t++)
                    tape_state = tape_exec(tape[t], tape_state, result ^ a, c ^ chain);
                result ^= tape_state;
            }

            uint32_t sel = rotl64(result,c) & 0xf;
            uint64_t v;
            if (sel>=10) v=stage3_heavy128(sel,a,b,c,result,r);
            else if (sel<=2) v=stage3_isqrt_heavy(sel,a,b,c,result,i,j);
            else switch(sel) {
                case 3:v=(a+b)*c;break; case 4:v=(b-c)*a;break; case 5:v=c-a+b;break;
                case 6:v=a-b+c;break; case 7:v=b*c+a;break; case 8:v=c*a+b;break;
                case 9:v=a*b*c;break; default:v=0;
            }

            uint64_t idx_seed = (v ^ chain) ^ result;
            result = rotl64(idx_seed, r);
            bool use_b = pick_half(v);
            uint64_t idx_t = map_index(idx_seed);
            uint64_t t2 = (use_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;
            uint64_t w_a = map_index(t2^result^0x9e3779b97f4a7c15ULL);
            uint64_t w_b = map_index(w_a^~result^0xd2b74407b1ce6e93ULL);
            uint64_t old_a = mem_buffer_a[w_a], old_b = mem_buffer_b[w_b];
            mem_buffer_b[w_b] = old_b ^ (old_a ^ rotr64(t2, i+j));
            mem_buffer_a[w_a] = t2;
        }
        addr_a = modular_power(addr_a, addr_b, result);
        addr_b = isqrt_v3(result) * ((uint64_t)r+1) * isqrt_v3(addr_a);
    }
    mem_buffer_a[0] = addr_a ^ addr_b ^ state[0] ^ tape_state;
}

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

    printf("XelisHash V4 Full — S-box + TapeMix + Dataset\n");
    printf("===============================================\n");
    printf("Device %d: %s (%d SMs, %.1f GB)\n", dev, prop.name,
           prop.multiProcessorCount, prop.totalGlobalMem/(1024.0*1024.0*1024.0));
    printf("Dataset: %d MB (%llu entries), %d reads/iter\n",
           DATASET_SIZE_MB, (unsigned long long)DATASET_SIZE, DATASET_READS);
    printf("Tape: len=%d freq=%d\n", TAPE_LEN, TAPE_FREQ);

    // Allocate and generate dataset
    size_t dataset_bytes = DATASET_SIZE * 8;
    uint64_t *d_dataset;
    printf("Allocating %d MB dataset...\n", DATASET_SIZE_MB);
    CUDA_CHECK(cudaMalloc(&d_dataset, dataset_bytes));
    { int t=256; int g=(int)((DATASET_SIZE+t-1)/t);
      generate_dataset<<<g,t>>>(d_dataset, DATASET_SIZE);
      CUDA_CHECK(cudaDeviceSynchronize()); }

    // Verify SUPRNOVA marker
    uint64_t marker;
    CUDA_CHECK(cudaMemcpy(&marker, d_dataset + 0x53, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printf("Dataset[0x53] = 0x%016llX (SUPRNOVA seed: 0x%016llX)\n",
           (unsigned long long)marker, (unsigned long long)SUPRNOVA_SEED);

    if (batch == 0) {
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        size_t per_hash = (size_t)MEMORY_SIZE * 8;
        batch = (int)((free_mem * 0.45) / per_hash);
        batch = (batch / 32) * 32;
        if (batch < 32) batch = 32;
    }

    size_t scratch_bytes = (size_t)batch * MEMORY_SIZE * 8;
    printf("Batch: %d hashes\n\n", batch);

    uint64_t *d_scratch;
    CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
    { uint64_t n=(uint64_t)batch*MEMORY_SIZE; int t=256,b2=(int)((n+t-1)/t);
      fill_random<<<b2,t>>>(d_scratch,n,0xCAFEBABE42ULL);
      CUDA_CHECK(cudaDeviceSynchronize()); }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int sb_tpb=32, sb_grid=(batch+sb_tpb-1)/sb_tpb;
    size_t sb_smem = (size_t)sb_tpb * SBOX_ENTRIES * sizeof(uint64_t);
    CUDA_CHECK(cudaFuncSetAttribute(xelis_stage3_full,
               cudaFuncAttributeMaxDynamicSharedMemorySize, sb_smem));

    // Warmup
    xelis_stage3_full<<<sb_grid,sb_tpb,sb_smem>>>(d_scratch, d_dataset, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float avg = 0;
    for (int ri = 0; ri < runs; ri++) {
        CUDA_CHECK(cudaEventRecord(start));
        xelis_stage3_full<<<sb_grid,sb_tpb,sb_smem>>>(d_scratch, d_dataset, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float khs = batch/(ms/1000.0f)/1000.0f;
        printf("  Run %d: %.2f ms (%.2f kH/s)\n", ri+1, ms, khs);
        if (ri > 0) avg += ms;
    }
    avg /= (runs > 1 ? runs-1 : 1);
    float khs = batch/(avg/1000.0f)/1000.0f;

    printf("\n===============================================\n");
    printf("RESULTS: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("V4-Full: %.2f kH/s (%.1f H/s/SM)\n", khs, khs*1000/prop.multiProcessorCount);
    printf("Dataset: %d MB, %d reads/iter | Tape: TL=%d TF=%d\n",
           DATASET_SIZE_MB, DATASET_READS, TAPE_LEN, TAPE_FREQ);
    printf("\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_dataset));
    CUDA_CHECK(cudaFree(d_scratch));
    return 0;
}
