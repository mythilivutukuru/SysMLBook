#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ─── tuneable constants ───────────────────────────────────────────────────────
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define TILE_M 16   // output tile rows  covered by one warp
#define TILE_N 16   // output tile cols  covered by one warp

// Warps per block (feel free to tune)
#define WARPS_PER_BLOCK_X 4
#define WARPS_PER_BLOCK_Y 2

// ─── CUDA error-check helper ─────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Helper kernel: convert a row-major FP32 matrix to FP16
// ─────────────────────────────────────────────────────────────────────────────
__global__ void convertFp32ToFp16(half *out, const float *in, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total)
        // TODO: convert in[idx] from float to half and write to out[idx]
        //   Hint: use __float2half()
        out[idx] = /* TODO */ __float2half(0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor Core matrix-multiplication kernel
//
// Computes C (FP32, M×N) = A (FP16, M×K) × B (FP16, K×N)
// Both A and B are row-major in global memory.
// Each warp owns one WMMA_M × WMMA_N output tile.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void matmulTensorCore(
    const half  *A,   // [M × K] row-major, FP16
    const half  *B,   // [K × N] row-major, FP16
    float       *C,   // [M × N] row-major, FP32  (output)
    int M, int N, int K)
{
    // ── identify which output tile this warp owns ──────────────────────────
    int warpId   = threadIdx.x / warpSize;                  // warp index within block
    int warpRow  = warpId / WARPS_PER_BLOCK_X;              // warp row within block
    int warpCol  = warpId % WARPS_PER_BLOCK_X;              // warp col within block

    // global tile position (in units of WMMA_M / WMMA_N)
    int tileRow  = blockIdx.y * WARPS_PER_BLOCK_Y + warpRow; // output tile row
    int tileCol  = blockIdx.x * WARPS_PER_BLOCK_X + warpCol; // output tile col

    // guard: some warps may be out-of-bounds for non-square matrices
    if (tileRow * WMMA_M >= M || tileCol * WMMA_N >= N) return;

    // ── declare WMMA fragments ─────────────────────────────────────────────

    // TODO: declare a_frag for matrix_a, shape WMMA_M×WMMA_N×WMMA_K,
    //       element type half, layout row_major
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;

    // TODO: declare b_frag for matrix_b, shape WMMA_M×WMMA_N×WMMA_K,
    //       element type half, layout row_major
    wmma::fragment</* TODO */, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

    // TODO: declare acc_frag as accumulator, shape WMMA_M×WMMA_N×WMMA_K, float
    wmma::fragment</* TODO */, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // TODO: zero-initialise acc_frag using wmma::fill_fragment
    /* TODO */

    // ── K-loop: iterate over K-tiles ───────────────────────────────────────
    for (int kTile = 0; kTile < K; kTile += WMMA_K)
    {
        // Compute pointers to the start of each warp's A and B tiles
        // in global memory for this K-iteration.
        //   a_ptr: row (tileRow * WMMA_M), column kTile  → stride between rows is K
        //   b_ptr: row kTile,              column (tileCol * WMMA_N) → stride between rows is N
        const half *a_ptr = /* TODO */ nullptr;
        const half *b_ptr = /* TODO */ nullptr;

        // TODO: load a_frag from global memory (a_ptr) with leading dimension K
        //   Hint: wmma::load_matrix_sync(frag, ptr, ldm)
        /* TODO */

        // TODO: load b_frag from global memory (b_ptr) with leading dimension N
        /* TODO */

        // TODO: perform the warp-level matrix-multiply-accumulate
        //   Hint: wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag)
        /* TODO */
    }

    // ── store the result tile back to global memory (FP32) ─────────────────

    // TODO: compute c_tile pointer — top-left of this warp's output tile in C
    //       i.e. C + (tileRow * WMMA_M) * N + (tileCol * WMMA_N)
    // TODO: store acc_frag to c_tile using
    //   wmma::store_matrix_sync(ptr, frag, ldm, wmma::mem_row_major)
    /* TODO */
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch wrapper
// ─────────────────────────────────────────────────────────────────────────────
void launchMatmul(const half *d_A, const half *d_B, float *d_C,
                  int M, int N, int K)
{
    // TODO: choose grid and block dimensions so that every 16×16 output tile
    //       is handled by exactly one warp.
    //
    //   Hints:
    //     - block has (WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y) warps
    //       → blockDim.x = that number * warpSize, blockDim.y = 1
    //     - gridDim covers all tiles in X (N direction) and Y (M direction)

    dim3 blockDim(/* TODO */);
    dim3 gridDim (/* TODO */);

    matmulTensorCore<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference implementation  (DO NOT MODIFY)
// ─────────────────────────────────────────────────────────────────────────────
void cpuMatmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[m * K + k] * B[k * N + n];
            C[m * N + n] = acc;
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test harness  (DO NOT MODIFY)
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int N = (argc > 2) ? atoi(argv[2]) : 512;
    int K = (argc > 3) ? atoi(argv[3]) : 512;

    printf("Matrix size: M=%d  N=%d  K=%d\n", M, N, K);

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *h_A   = (float *)malloc(bytesA);
    float *h_B   = (float *)malloc(bytesB);
    float *h_C   = (float *)malloc(bytesC);
    float *h_Ref = (float *)malloc(bytesC);

    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = (rand() % 10 - 5) / 5.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (rand() % 10 - 5) / 5.0f;

    printf("Running CPU reference...\n");
    clock_t t0 = clock();
    cpuMatmul(h_A, h_B, h_Ref, M, N, K);
    double cpuMs = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;
    printf("CPU reference time      : %.1f ms\n", cpuMs);

    float *d_A_f32, *d_B_f32;
    half  *d_A_f16, *d_B_f16;
    float *d_C;

    CUDA_CHECK(cudaMalloc(&d_A_f32, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B_f32, bytesB));
    CUDA_CHECK(cudaMalloc(&d_A_f16, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_f16, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C,     bytesC));

    CUDA_CHECK(cudaMemcpy(d_A_f32, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_f32, h_B, bytesB, cudaMemcpyHostToDevice));

    int threads = 256;
    convertFp32ToFp16<<<(M*K+threads-1)/threads, threads>>>(d_A_f16, d_A_f32, M, K);
    convertFp32ToFp16<<<(K*N+threads-1)/threads, threads>>>(d_B_f16, d_B_f32, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaMemset(d_C, 0, bytesC));
    CUDA_CHECK(cudaEventRecord(ev0));
    launchMatmul(d_A_f16, d_B_f16, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float gpuMs = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, ev0, ev1));
    printf("Tensor Core kernel time : %.2f ms\n", gpuMs);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    float maxErr = 0.f;
    for (int i = 0; i < M * N; ++i)
        maxErr = fmaxf(maxErr, fabsf(h_C[i] - h_Ref[i]));

    const float threshold = 0.5f;
    printf("Max absolute error      : %f   (threshold %.1f) --> %s\n",
           maxErr, threshold, maxErr <= threshold ? "PASS" : "FAIL");

    free(h_A); free(h_B); free(h_C); free(h_Ref);
    cudaFree(d_A_f32); cudaFree(d_B_f32);
    cudaFree(d_A_f16); cudaFree(d_B_f16);
    cudaFree(d_C);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
