#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* ─────────────────────────── configuration ─────────────────────────── */

#define N          1024
#define BLOCK_SIZE   16
#define FLOAT_TOL  1e-3f

/* ───────────────────────────── utilities ───────────────────────────── */

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t st = (call);                                         \
        if (st != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d — code %d\n",           \
                    __FILE__, __LINE__, (int)st);                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

typedef struct { cudaEvent_t start, stop; } GpuTimer;

void timer_start(GpuTimer *t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
    cudaEventRecord(t->start);
}
float timer_stop(GpuTimer *t) {
    float ms;
    cudaEventRecord(t->stop);
    cudaEventSynchronize(t->stop);
    cudaEventElapsedTime(&ms, t->start, t->stop);
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
    return ms;
}

void rand_matrix(float *M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}

int check_result(const float *ref, const float *gpu, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabsf(ref[i] - gpu[i]) > FLOAT_TOL) {
            printf("  MISMATCH at [%d,%d]: ref=%.6f  gpu=%.6f\n",
                   i / n, i % n, ref[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * TASK 1 — GPU Kernel  (SOLUTION)
 * ═══════════════════════════════════════════════════════════════════════ */
__global__ void matmul_gpu(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    /* Discard threads that map outside the matrix */
    if (row >= n || col >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * B[k * n + col];

    C[row * n + col] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * TASK 2 — cuBLAS  (SOLUTION)
 * ═══════════════════════════════════════════════════════════════════════ */
void matmul_cublas(cublasHandle_t handle,
                   const float *d_A, const float *d_B, float *d_C, int n)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  /* no transpose on either operand        */
        n, n, n,                   /* m, n, k — all N for square matrices   */
        &alpha,
        d_B, n,                    /* first operand  = B (row-major → Bᵀ)   */
        d_A, n,                    /* second operand = A (row-major → Aᵀ)   */
        &beta,
        d_C, n                     /* output C, leading dimension = n        */
    ));
}

/* ═══════════════════════════════════════════════════════════════════════
 * CPU Reference
 * ═══════════════════════════════════════════════════════════════════════ */
void matmul_cpu(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

/* ═══════════════════════════════════════════════════════════════════════
 * main 
 * ═══════════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("Matrix size: %d x %d\n\n", N, N);
    srand(42);

    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A   = (float*)malloc(bytes);
    float *h_B   = (float*)malloc(bytes);
    float *h_C   = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    rand_matrix(h_A, N);
    rand_matrix(h_B, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    printf("Running CPU reference (this may take a while)...\n");
    clock_t t0 = clock();
    matmul_cpu(h_A, h_B, h_ref, N);
    double cpu_ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;
    printf("[CPU ref]      Time: %.1f ms\n\n", cpu_ms);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Task 1: custom GPU kernel */
    {
        GpuTimer t; timer_start(&t);
        matmul_gpu<<<grid, block>>>(d_A, d_B, d_C, N);
        float ms = timer_stop(&t);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        printf("[GPU kernel]   Time: %7.2f ms  | %s\n",
               ms, check_result(h_ref, h_C, N) ? "CORRECT" : "WRONG");
    }

    /* Task 2: cuBLAS */
    {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        GpuTimer t; timer_start(&t);
        matmul_cublas(handle, d_A, d_B, d_C, N);
        float ms = timer_stop(&t);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        printf("[cuBLAS]       Time: %7.2f ms  | %s\n",
               ms, check_result(h_ref, h_C, N) ? "CORRECT" : "WRONG");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);

    return 0;
}
