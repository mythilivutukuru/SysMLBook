#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * Network dimensions
 * ───────────────────────────────────────────────────────────────────────────── */
#define N     64
#define D_IN  32
#define D_H   64
#define D_OUT 16

/* ─────────────────────────────────────────────────────────────────────────────
 * Error-checking macro
 * ───────────────────────────────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d  →  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ═════════════════════════════════════════════════════════════════════════════
 * KERNEL 1 — Fused Matrix-Multiply + ReLU
 *
 * C[i][j] = ReLU( sum_k A[i][k] * B[k][j]  +  bias[j] )
 * ═════════════════════════════════════════════════════════════════════════════*/
__global__ void matmul_relu_kernel(const float *A,
                                   const float *B,
                                   const float *bias,
                                         float *C,
                                   int rows_A,
                                   int inner,
                                   int cols_B)
{
    /* TODO 1 — Row and column indices */
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    /* TODO 2 — Bounds guard */
    if (row >= rows_A || col >= cols_B) return;

    /* TODO 3 — Dot product */
    float acc = 0.0f;
    for (int k = 0; k < inner; ++k)
        acc += A[row * inner + k] * B[k * cols_B + col];

    /* TODO 4 — Bias */
    acc += bias[col];

    /* TODO 5 — ReLU and write */
    C[row * cols_B + col] = fmaxf(acc, 0.0f);
}


/* ═════════════════════════════════════════════════════════════════════════════
 * KERNEL 2 — Plain Matrix-Multiply (no activation)
 *
 * C[i][j] = sum_k A[i][k] * B[k][j]  +  bias[j]
 * ═════════════════════════════════════════════════════════════════════════════*/
__global__ void matmul_kernel(const float *A,
                              const float *B,
                              const float *bias,
                                    float *C,
                              int rows_A,
                              int inner,
                              int cols_B)
{
    /* TODO 6 — Row and column indices */
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    /* TODO 7 — Bounds guard */
    if (row >= rows_A || col >= cols_B) return;

    /* TODO 8 — Dot product */
    float acc = 0.0f;
    for (int k = 0; k < inner; ++k)
        acc += A[row * inner + k] * B[k * cols_B + col];

    /* TODO 9 — Bias, write (no ReLU) */
    C[row * cols_B + col] = acc + bias[col];
}


/* ═════════════════════════════════════════════════════════════════════════════
 * CPU REFERENCE IMPLEMENTATION
 * ═════════════════════════════════════════════════════════════════════════════*/
static void mlp_cpu(const float *X,
                    const float *W1, const float *b1,
                    const float *W2, const float *b2,
                          float *H,
                          float *Y)
{
    /* Layer 1: H = ReLU(X @ W1 + b1) */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D_H; ++j) {
            float acc = b1[j];
            for (int k = 0; k < D_IN; ++k)
                acc += X[i * D_IN + k] * W1[k * D_H + j];
            H[i * D_H + j] = acc > 0.0f ? acc : 0.0f;
        }
    }

    /* Layer 2: Y = H @ W2 + b2 */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D_OUT; ++j) {
            float acc = b2[j];
            for (int k = 0; k < D_H; ++k)
                acc += H[i * D_H + k] * W2[k * D_OUT + j];
            Y[i * D_OUT + j] = acc;
        }
    }
}


/* ═════════════════════════════════════════════════════════════════════════════
 * GPU LAUNCH WRAPPER
 * ═════════════════════════════════════════════════════════════════════════════*/
static void mlp_forward_gpu(const float *h_X,
                             const float *h_W1, const float *h_b1,
                             const float *h_W2, const float *h_b2,
                                   float *h_Y,
                                   float *d_X,
                                   float *d_W1, float *d_b1,
                                   float *d_W2, float *d_b2,
                                   float *d_H,
                                   float *d_Y)
{
    /* TODO 10 — Create stream */
    cudaStream_t stream;
    CUDA_CHECK( cudaStreamCreate(&stream) );

    /* TODO 11-15 — Async H2D copies */
    CUDA_CHECK( cudaMemcpyAsync(d_X,  h_X,
                                N * D_IN  * sizeof(float),
                                cudaMemcpyHostToDevice, stream) );

    CUDA_CHECK( cudaMemcpyAsync(d_W1, h_W1,
                                D_IN * D_H * sizeof(float),
                                cudaMemcpyHostToDevice, stream) );

    CUDA_CHECK( cudaMemcpyAsync(d_b1, h_b1,
                                D_H * sizeof(float),
                                cudaMemcpyHostToDevice, stream) );

    CUDA_CHECK( cudaMemcpyAsync(d_W2, h_W2,
                                D_H * D_OUT * sizeof(float),
                                cudaMemcpyHostToDevice, stream) );

    CUDA_CHECK( cudaMemcpyAsync(d_b2, h_b2,
                                D_OUT * sizeof(float),
                                cudaMemcpyHostToDevice, stream) );

    /* TODO 16 — Launch kernel 1: hidden layer H = ReLU(X @ W1 + b1) */
    dim3 block1(16, 16);
    dim3 grid1( (D_H  + block1.x - 1) / block1.x,
                (N    + block1.y - 1) / block1.y );
    matmul_relu_kernel<<<grid1, block1, 0, stream>>>(
        d_X, d_W1, d_b1, d_H, N, D_IN, D_H);
    CUDA_CHECK( cudaGetLastError() );

    /* TODO 17 — Launch kernel 2: output layer Y = H @ W2 + b2 */
    dim3 block2(16, 16);
    dim3 grid2( (D_OUT + block2.x - 1) / block2.x,
                (N    + block2.y - 1) / block2.y );
    matmul_kernel<<<grid2, block2, 0, stream>>>(
        d_H, d_W2, d_b2, d_Y, N, D_H, D_OUT);
    CUDA_CHECK( cudaGetLastError() );

    /* TODO 18 — Async D2H copy of Y */
    CUDA_CHECK( cudaMemcpyAsync(h_Y, d_Y,
                                N * D_OUT * sizeof(float),
                                cudaMemcpyDeviceToHost, stream) );

    /* TODO 19 — Synchronise and destroy stream */
    CUDA_CHECK( cudaStreamSynchronize(stream) );
    CUDA_CHECK( cudaStreamDestroy(stream) );
}


/* ═════════════════════════════════════════════════════════════════════════════
 * MAIN 
 * ═════════════════════════════════════════════════════════════════════════════*/
int main(void)
{
    size_t sz_X   = N * D_IN  * sizeof(float);
    size_t sz_W1  = D_IN * D_H  * sizeof(float);
    size_t sz_b1  = D_H         * sizeof(float);
    size_t sz_W2  = D_H  * D_OUT * sizeof(float);
    size_t sz_b2  = D_OUT        * sizeof(float);
    size_t sz_H   = N * D_H   * sizeof(float);
    size_t sz_Y   = N * D_OUT  * sizeof(float);

    float *h_X,  *h_W1, *h_b1, *h_W2, *h_b2, *h_H_cpu, *h_Y_cpu, *h_Y_gpu;
    CUDA_CHECK( cudaMallocHost(&h_X,     sz_X)  );
    CUDA_CHECK( cudaMallocHost(&h_W1,    sz_W1) );
    CUDA_CHECK( cudaMallocHost(&h_b1,    sz_b1) );
    CUDA_CHECK( cudaMallocHost(&h_W2,    sz_W2) );
    CUDA_CHECK( cudaMallocHost(&h_b2,    sz_b2) );
    CUDA_CHECK( cudaMallocHost(&h_H_cpu, sz_H)  );
    CUDA_CHECK( cudaMallocHost(&h_Y_cpu, sz_Y)  );
    CUDA_CHECK( cudaMallocHost(&h_Y_gpu, sz_Y)  );

    srand(42);
    for (int i = 0; i < N * D_IN;   ++i) h_X [i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < D_IN * D_H;  ++i) h_W1[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < D_H;          ++i) h_b1[i] = (rand() % 40  -  20) / 100.0f;
    for (int i = 0; i < D_H * D_OUT;  ++i) h_W2[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < D_OUT;         ++i) h_b2[i] = (rand() % 40  -  20) / 100.0f;

    mlp_cpu(h_X, h_W1, h_b1, h_W2, h_b2, h_H_cpu, h_Y_cpu);
    printf("[CPU] MLP forward pass complete.\n");

    float *d_X, *d_W1, *d_b1, *d_W2, *d_b2, *d_H, *d_Y;
    CUDA_CHECK( cudaMalloc(&d_X,  sz_X)  );
    CUDA_CHECK( cudaMalloc(&d_W1, sz_W1) );
    CUDA_CHECK( cudaMalloc(&d_b1, sz_b1) );
    CUDA_CHECK( cudaMalloc(&d_W2, sz_W2) );
    CUDA_CHECK( cudaMalloc(&d_b2, sz_b2) );
    CUDA_CHECK( cudaMalloc(&d_H,  sz_H)  );
    CUDA_CHECK( cudaMalloc(&d_Y,  sz_Y)  );

    mlp_forward_gpu(h_X,  h_W1, h_b1, h_W2, h_b2,
                    h_Y_gpu,
                    d_X,  d_W1, d_b1, d_W2, d_b2, d_H, d_Y);
    printf("[GPU] MLP forward pass complete.\n");

    printf("Checking correctness...\n");
    float max_err = 0.0f;
    int   fail_i  = -1;
    for (int i = 0; i < N * D_OUT; ++i) {
        float err = fabsf(h_Y_cpu[i] - h_Y_gpu[i]);
        if (err > max_err) { max_err = err; }
        if (err > 1e-3f && fail_i < 0) fail_i = i;
    }
    printf("Max absolute error: %f\n", max_err);
    if (fail_i < 0) {
        printf("PASS: GPU output matches CPU reference.\n");
    } else {
        printf("FAIL: first mismatch at index %d  cpu=%.6f  gpu=%.6f\n",
               fail_i, h_Y_cpu[fail_i], h_Y_gpu[fail_i]);
    }

    cudaFree(d_X);  cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_H);  cudaFree(d_Y);
    cudaFreeHost(h_X);  cudaFreeHost(h_W1); cudaFreeHost(h_b1);
    cudaFreeHost(h_W2); cudaFreeHost(h_b2);
    cudaFreeHost(h_H_cpu); cudaFreeHost(h_Y_cpu); cudaFreeHost(h_Y_gpu);
    return 0;
}
