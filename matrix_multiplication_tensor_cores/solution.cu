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

#define TILE_M 16
#define TILE_N 16

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
        out[idx] = __float2half(in[idx]);   // ← TODO filled
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor Core matrix-multiplication kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void matmulTensorCore(
    const half *A, const half *B, float *C,
    int M, int N, int K)
{
    int warpId  = threadIdx.x / warpSize;
    int warpRow = warpId / WARPS_PER_BLOCK_X;
    int warpCol = warpId % WARPS_PER_BLOCK_X;

    int tileRow = blockIdx.y * WARPS_PER_BLOCK_Y + warpRow;
    int tileCol = blockIdx.x * WARPS_PER_BLOCK_X + warpCol;

    if (tileRow * WMMA_M >= M || tileCol * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                 acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int kTile = 0; kTile < K; kTile += WMMA_K)
    {
        // Load directly from global memory — stride is the full matrix width
        const half *a_ptr = A + (tileRow * WMMA_M) * K + kTile;
        const half *b_ptr = B + kTile * N + (tileCol * WMMA_N);

        wmma::load_matrix_sync(a_frag, a_ptr, K);   // row stride = K
        wmma::load_matrix_sync(b_frag, b_ptr, N);   // row stride = N

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float *c_tile = C + (tileRow * WMMA_M) * N + (tileCol * WMMA_N);
    wmma::store_matrix_sync(c_tile, acc_frag, N, wmma::mem_row_major);
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch wrapper
// ─────────────────────────────────────────────────────────────────────────────
void launchMatmul(const half *d_A, const half *d_B, float *d_C,
                  int M, int N, int K)
{
    // TODO filled:
    //   - one warp per output tile
    //   - block = WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y warps → ×32 threads
    int warpsPerBlock = WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y;
    dim3 blockDim(warpsPerBlock * 32, 1, 1);

    // grid covers all tiles
    dim3 gridDim(
        (N + WMMA_N * WARPS_PER_BLOCK_X - 1) / (WMMA_N * WARPS_PER_BLOCK_X),
        (M + WMMA_M * WARPS_PER_BLOCK_Y - 1) / (WMMA_M * WARPS_PER_BLOCK_Y),
        1);

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
