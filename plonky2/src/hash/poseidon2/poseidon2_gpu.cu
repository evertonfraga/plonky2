// poseidon2_gpu.cu — Poseidon2 for plonky2 with exact plonky2 constants.
// Goldilocks: p = 2^64 - 2^32 + 1, WIDTH=12, ROUNDS_F=8, ROUNDS_P=22, alpha=7
#include <stdint.h>
#include <cuda_runtime.h>
#define WIDTH 12
#define ROUNDS_F_HALF 4
#define ROUNDS_P 22
#define NUM_HASH_OUT 4
#define GL_EPS 0xFFFFFFFFULL

__device__ __constant__ uint64_t EXT_CONSTS[] = {
    0xd70193d17ab3b7d6ULL, 0xa2c3662a78a9162bULL, 0x7a9fda827556ad44ULL, 0xe8d5501818c99643ULL, 0x4c7a8fced4d5fd38ULL, 0x55ab38985c0c513dULL,
    0x28a17bd016210b0bULL, 0x8f8277679ec32fa8ULL, 0x768b3c3d68a460e9ULL, 0x872a022eb559d941ULL, 0xd1316dd4b3b97973ULL, 0xa7b608e578321000ULL,
    0x3fa02c87b0bee026ULL, 0x7a38f0022e13c31eULL, 0x00c054f3c5e8d20dULL, 0x439f50f4bca7242fULL, 0x4d0938aa57cd517fULL, 0xb2e03ac5fb6b9a7dULL,
    0xe29d1f4237bedca8ULL, 0x05b7c844bc99b848ULL, 0x91cc0b73f34e17edULL, 0x876e4427694bd755ULL, 0x67002ae0725c612dULL, 0x05351f20e0b6315fULL,
    0x2e3b9ef5457eb60bULL, 0xd9ac17618c3783ddULL, 0x0807528ad8874bcfULL, 0xc78d546a455d2a0eULL, 0xf8b930c81e2481f0ULL, 0x712707d8dff3b041ULL,
    0xdcb8c0aa0b9d34c3ULL, 0x9baddbdf2ee3a468ULL, 0x2dd16d50c5176c78ULL, 0x89eac5cfbc075cd3ULL, 0x2a741dea181587f3ULL, 0x1a4d6aa85a113d84ULL,
    0x4d736286a2387e34ULL, 0x8bad5dfc4fcb3ee3ULL, 0x84fbd03adb77c56aULL, 0x8d5cdd1a23ec53a2ULL, 0x036f08f08fff28ecULL, 0xb717a3f4dbdfb443ULL,
    0x58a074b5509d645cULL, 0xf92bf834e4b87718ULL, 0x1541c3a0baa5ac4bULL, 0x22149e6783e67692ULL, 0x9be8b5d9e112476fULL, 0x41e0969f62babb76ULL,
    0xbc585ad3b9443dbbULL, 0xf28dd3206975cbb1ULL, 0xdd8815e53ca045e0ULL, 0xde82c416b9e701baULL, 0xc5cb875233afa025ULL, 0x7212697cd897ffa9ULL,
    0x67844790aa63cfd7ULL, 0xdc0b9cfa97fe65c3ULL, 0xe8fe091869a82070ULL, 0x62902bb2e413c6d1ULL, 0x29f9f5001fb84f57ULL, 0xbe1014796ef5f8beULL,
    0x71feb53e9bdba19cULL, 0x251054f592ebb71cULL, 0xe1a57643a4bb284bULL, 0xa4ba6f87a45b739bULL, 0x2c1fcade0b958c49ULL, 0xbbb424cda9a3e360ULL,
    0x2ca647354c5f3f54ULL, 0xc9277b64d152e084ULL, 0xdbc9ac97445eff17ULL, 0x6f6cdf3198969f70ULL, 0x1de29d14fa76d8f1ULL, 0x73337458a8cc1d19ULL,
    0xb87e775e2fb3ab23ULL, 0xf166a1c7a565c80bULL, 0xb24be06f426c747fULL, 0xc281e8c49482ce00ULL, 0x51974c3b3b726c2dULL, 0x87444cf8caf7d619ULL,
    0x7c362f827a580cedULL, 0x9567af14667647a0ULL, 0xcbf0473cbec54e37ULL, 0xe3209dedeff4f620ULL, 0xd43ad94e45a4c4eeULL, 0x976981ee73f41768ULL,
    0xef707a224e207258ULL, 0x2fc779e10e6362eeULL, 0x29b5ee60ad8c891fULL, 0x96b37b39d8bfd667ULL, 0x877df68a8b22e733ULL, 0x5c41746f562c8d9fULL,
    0x0c9d76751052b71aULL, 0xfb3465341bf1c087ULL, 0xa0d14dc614d15eb1ULL, 0xdc27d17136906fa6ULL, 0x482e163b05ec397fULL, 0x0273a462992366efULL,
};
__device__ __constant__ uint64_t INT_CONSTS[] = {
    0xa571418d95897b60ULL, 0x8f32676574fcf6d3ULL, 0x731102d4e3fb1bbeULL, 0x0330f08328a82d2bULL,
    0x7f0449b6557f785dULL, 0x62f06210658dcbcbULL, 0xd5a98af9f89c458bULL, 0x77ec69083a346385ULL,
    0xef7ca48bbc27f890ULL, 0x53e9652f61eac532ULL, 0xa71c634abff4f0ccULL, 0xb16f5f0d7e28ea29ULL,
    0xc9dde31d0a003ab2ULL, 0x2ddadf9775902533ULL, 0xe4fa73fb16408b47ULL, 0x90242ebc00d2ee59ULL,
    0xbb02dffd9f381982ULL, 0xdea328364c50907cULL, 0x1395d3b924857cf8ULL, 0x7d3ead0d5aec04e6ULL,
    0xc2f12be3fed74668ULL, 0x0ba3c338f8c3d285ULL,
};
__device__ __constant__ uint64_t DIAG[] = {
    0xc3b6c08e23ba9300ULL, 0xd84b5de94a324fb6ULL, 0x0d0c371c5b35b84fULL, 0x7964f570e7188037ULL,
    0x5daf18bbd996604bULL, 0x6743bc47b9595257ULL, 0x5528b9362c59bb70ULL, 0xac45e25b7127b68bULL,
    0xa2077d7dfbb606b5ULL, 0xf3faac6faee378aeULL, 0x0c6388b51545e883ULL, 0xd27dbb6944917b60ULL,
};

__device__ __forceinline__ uint64_t gl_reduce128(unsigned __int128 x) {
    uint64_t lo = (uint64_t)x, hi = (uint64_t)(x >> 64);
    uint64_t hi_hi = hi >> 32, hi_lo = hi & GL_EPS;
    uint64_t t0 = lo - hi_hi;
    if (lo < hi_hi) t0 -= GL_EPS;
    unsigned __int128 t1 = (unsigned __int128)hi_lo * GL_EPS;
    unsigned __int128 t2 = (unsigned __int128)t0 + t1;
    return (uint64_t)t2 + (uint64_t)(t2 >> 64) * GL_EPS;
}
__device__ __forceinline__ uint64_t gl_add(uint64_t a, uint64_t b) {
    unsigned __int128 s = (unsigned __int128)a + b;
    return (uint64_t)s + (uint64_t)(s >> 64) * GL_EPS;
}
__device__ __forceinline__ uint64_t gl_mul(uint64_t a, uint64_t b) {
    return gl_reduce128((unsigned __int128)a * b);
}
__device__ __forceinline__ uint64_t gl_mac(uint64_t acc, uint64_t x, uint64_t y) {
    return gl_reduce128((unsigned __int128)acc + (unsigned __int128)x * y);
}
__device__ __forceinline__ uint64_t gl_sbox(uint64_t x) {
    uint64_t x2 = gl_mul(x,x), x4 = gl_mul(x2,x2);
    return gl_mul(gl_mul(x,x2), x4);
}

__device__ void ext_linear(uint64_t s[WIDTH]) {
    #pragma unroll
    for (int g = 0; g < 3; g++) {
        int b = g*4;
        uint64_t t01 = gl_add(s[b],s[b+1]), t23 = gl_add(s[b+2],s[b+3]);
        uint64_t t = gl_add(t01,t23), x0=s[b], x2=s[b+2];
        s[b]   = gl_add(gl_add(t,t01),s[b+1]);
        s[b+1] = gl_add(gl_add(t,s[b+1]),gl_add(x2,x2));
        s[b+2] = gl_add(gl_add(t,t23),s[b+3]);
        s[b+3] = gl_add(gl_add(t,s[b+3]),gl_add(x0,x0));
    }
    uint64_t sums[4];
    #pragma unroll
    for (int k=0;k<4;k++) sums[k]=gl_add(gl_add(s[k],s[k+4]),s[k+8]);
    #pragma unroll
    for (int i=0;i<WIDTH;i++) s[i]=gl_add(s[i],sums[i%4]);
}

__device__ void int_linear(uint64_t s[WIDTH]) {
    unsigned __int128 acc=0;
    #pragma unroll
    for (int i=0;i<WIDTH;i++) acc+=s[i];
    uint64_t lo=(uint64_t)acc, hi=(uint64_t)(acc>>64);
    uint64_t sum=lo+hi*GL_EPS; if(sum<lo) sum+=GL_EPS;
    #pragma unroll
    for (int i=0;i<WIDTH;i++) s[i]=gl_mac(sum,s[i],DIAG[i]);
}

__device__ void permute(uint64_t s[WIDTH]) {
    ext_linear(s);
    #pragma unroll
    for (int r=0;r<ROUNDS_F_HALF;r++) {
        #pragma unroll
        for (int i=0;i<WIDTH;i++) s[i]=gl_add(s[i],EXT_CONSTS[r*WIDTH+i]);
        #pragma unroll
        for (int i=0;i<WIDTH;i++) s[i]=gl_sbox(s[i]);
        ext_linear(s);
    }
    #pragma unroll
    for (int r=0;r<ROUNDS_P;r++) {
        s[0]=gl_add(s[0],INT_CONSTS[r]); s[0]=gl_sbox(s[0]); int_linear(s);
    }
    #pragma unroll
    for (int r=ROUNDS_F_HALF;r<ROUNDS_F_HALF*2;r++) {
        #pragma unroll
        for (int i=0;i<WIDTH;i++) s[i]=gl_add(s[i],EXT_CONSTS[r*WIDTH+i]);
        #pragma unroll
        for (int i=0;i<WIDTH;i++) s[i]=gl_sbox(s[i]);
        ext_linear(s);
    }
}

extern "C" __global__ void poseidon2_two_to_one_batch(
    const uint64_t* __restrict__ in, uint64_t* __restrict__ out, uint32_t batch) {
    uint32_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=batch) return;
    uint64_t s[WIDTH];
    #pragma unroll
    for (int i=0;i<8;i++) s[i]=in[idx*8+i];
    #pragma unroll
    for (int i=8;i<WIDTH;i++) s[i]=0;
    permute(s);
    #pragma unroll
    for (int i=0;i<NUM_HASH_OUT;i++) out[idx*NUM_HASH_OUT+i]=s[i];
}

extern "C" __global__ void poseidon2_hash_no_pad_batch(
    const uint64_t* __restrict__ in, uint64_t* __restrict__ out,
    uint32_t batch, uint32_t leaf_len) {
    uint32_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=batch) return;
    const uint64_t* leaf = in + idx*leaf_len;
    uint64_t s[WIDTH] = {0};
    uint32_t pos=0;
    while (pos+8<=leaf_len) {
        #pragma unroll
        for (int i=0;i<8;i++) s[i]=gl_add(s[i],leaf[pos+i]);
        permute(s); pos+=8;
    }
    for (uint32_t i=0;i<leaf_len-pos;i++) s[i]=gl_add(s[i],leaf[pos+i]);
    if (leaf_len-pos>0) permute(s);
    #pragma unroll
    for (int i=0;i<NUM_HASH_OUT;i++) out[idx*NUM_HASH_OUT+i]=s[i];
}

extern "C" {
int poseidon2_two_to_one_gpu(const uint64_t* in_h, uint64_t* out_h, uint32_t batch) {
    uint64_t *d_in, *d_out;
    if (cudaMalloc(&d_in, batch*8*8)!=cudaSuccess) return -1;
    if (cudaMalloc(&d_out,batch*4*8)!=cudaSuccess) { cudaFree(d_in); return -1; }
    cudaMemcpy(d_in,in_h,batch*8*8,cudaMemcpyHostToDevice);
    poseidon2_two_to_one_batch<<<(batch+255)/256,256>>>(d_in,d_out,batch);
    cudaMemcpy(out_h,d_out,batch*4*8,cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
int poseidon2_hash_no_pad_gpu(const uint64_t* in_h, uint64_t* out_h, uint32_t batch, uint32_t leaf_len) {
    uint64_t *d_in, *d_out;
    if (cudaMalloc(&d_in, (size_t)batch*leaf_len*8)!=cudaSuccess) return -1;
    if (cudaMalloc(&d_out,(size_t)batch*4*8)!=cudaSuccess) { cudaFree(d_in); return -1; }
    cudaMemcpy(d_in,in_h,(size_t)batch*leaf_len*8,cudaMemcpyHostToDevice);
    poseidon2_hash_no_pad_batch<<<(batch+255)/256,256>>>(d_in,d_out,batch,leaf_len);
    cudaMemcpy(out_h,d_out,(size_t)batch*4*8,cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
}
