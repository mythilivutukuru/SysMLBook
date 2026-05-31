#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * Network dimensions
 * ───────────────────────────────────────────────────────────────────────────── */
#define N     64      /* batch size                  */
#define D_IN  32      /* input feature dimension     */
#define D_H   64      /* hidden layer width          */
#define D_OUT 16      /* output dimension            */

/* ─────────────────────────────────────────────────────────────────────────────
 * Convenience macro — wraps every CUDA call and prints file/line on error
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
 * Computes:  C[i][j] = ReLU( sum_k A[i][k] * B[k][j]  +  bias[j] )
 *
 * Arguments:
 *   A      — input  matrix, row-major, shape (rows_A, inner)
 *   B      — weight matrix, row-major, shape (inner,  cols_B)
 *   bias   — bias vector,               shape (cols_B,)
 *   C      — output matrix, row-major,  shape (rows_A, cols_B)
 *   rows_A — number of rows in A  (= N for the first layer)
 *   inner  — shared/contracted dimension (= D_IN for the first layer)
 *   cols_B — number of columns in B (= D_H for the first layer)
 *
 * Thread mapping (suggested):
 *   threadIdx.x + blockIdx.x * blockDim.x  →  column index j of C
 *   threadIdx.y + blockIdx.y * blockDim.y  →  row    index i of C
 * ═════════════════════════════════════════════════════════════════════════════*/
__global__ void matmul_relu_kernel(const float *A,
                                   const float *B,
                                   const float *bias,
                                         float *C,
                                   int rows_A,
                                   int inner,
                                   int cols_B)
{
    /* TODO 1 — Compute the row and column index this thread is responsible for.
     *
     *   int row = ...;
     *   int col = ...;
     */

    /* TODO 2 — Guard: return immediately if (row, col) is out of bounds.
     */

    /* TODO 3 — Accumulate the dot product over the inner dimension.
     */

    /* TODO 4 — Add the bias term.
     */

    /* TODO 5 — Apply ReLU activation and write the result to C.
     */
}


/* ═════════════════════════════════════════════════════════════════════════════
 * KERNEL 2 — Plain Matrix-Multiply (no activation)
 *
 * Computes:  C[i][j] = sum_k A[i][k] * B[k][j]  +  bias[j]
 *
 * Arguments: same layout as matmul_relu_kernel but WITHOUT ReLU.
 * ═════════════════════════════════════════════════════════════════════════════*/
__global__ void matmul_kernel(const float *A,
                              const float *B,
                              const float *bias,
                                    float *C,
                              int rows_A,
                              int inner,
                              int cols_B)
{
    /* TODO 6 — Compute row and column indices (same as kernel 1). */

    /* TODO 7 — Bounds check (same as kernel 1). */

    /* TODO 8 — Accumulate dot product (same as kernel 1). */

    /* TODO 9 — Add bias and write result (NO ReLU this time).
     */
}


/* ═════════════════════════════════════════════════════════════════════════════
 * CPU REFERENCE IMPLEMENTATION — do not modify
 *
 * Used to verify your GPU output.  Runs the exact same math on the host.
 * ═════════════════════════════════════════════════════════════════════════════*/
static void mlp_cpu(const float *X,
                    const float *W1, const float *b1,
                    const float *W2, const float *b2,
                          float *H,        /* (N, D_H)   intermediate */
                          float *Y)        /* (N, D_OUT) final output */
{
    /* Layer 1: H = ReLU(X @ W1 + b1) */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D_H; ++j) {
            float acc = b1[j];
            for (int k = 0; k < D_IN; ++k)
                acc += X[i * D_IN + k] * W1[k * D_H + j];
            H[i * D_H + j] = acc > 0.0f ? acc : 0.0f;  /* ReLU */
        }
    }

    /* Layer 2: Y = H @ W2 + b2 (no activation) */
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
 *
 * Orchestrates:
 *   1. Stream creation
 *   2. Async H2D copies of all inputs
 *   3. Kernel 1 launch  →  produces H on device
 *   4. Kernel 2 launch  →  produces Y on device
 *   5. Async D2H copy   →  brings Y back to host
 *   6. Stream synchronisation + destruction
 *
 * All device buffers are pre-allocated by main() and passed in.
 * ═════════════════════════════════════════════════════════════════════════════*/
static void mlp_forward_gpu(/* host pointers (read-only inputs) */
                             const float *h_X,
                             const float *h_W1, const float *h_b1,
                             const float *h_W2, const float *h_b2,
                             /* host output buffer (written by D2H copy) */
                                   float *h_Y,
                             /* pre-allocated device buffers */
                                   float *d_X,
                                   float *d_W1, float *d_b1,
                                   float *d_W2, float *d_b2,
                                   float *d_H,
                                   float *d_Y)
{
    /* ── Stream ─────────────────────────────────────────────────────────────── */

    /* TODO 10 — Declare and create a CUDA stream.
     *
     *   cudaStream_t stream;
     *   CUDA_CHECK( cudaStreamCreate(&stream) );
     */

    /* ── Async Host → Device copies ─────────────────────────────────────────── */
    /*
     * Use cudaMemcpyAsync (NOT cudaMemcpy) so the copy is enqueued into the
     * stream and can overlap with other work.
     *
     * Signature:
     *   cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream)
     */

    /* TODO 11 — Copy X    : N * D_IN  floats */
    /* TODO 12 — Copy W1   : D_IN * D_H floats */
    /* TODO 13 — Copy b1   : D_H floats */
    /* TODO 14 — Copy W2   : D_H * D_OUT floats */
    /* TODO 15 — Copy b2   : D_OUT floats */

    /* ── Kernel 1: Hidden layer ─────────────────────────────────────────────── */
    /*
     * Map 2-D output H (N × D_H) onto a 2-D grid of 2-D thread blocks.
     *
     * Suggested block size: 16 × 16 threads  (blockDim.x = 16, blockDim.y = 16)
     * Grid size must cover every element — use ceiling division:
     *   gridDim.x = (D_H  + blockDim.x - 1) / blockDim.x
     *   gridDim.y = (N    + blockDim.y - 1) / blockDim.y
     */

    /* TODO 16 — Set up dim3 block and dim3 grid, then launch matmul_relu_kernel
     *           into your stream.
     *
     *   dim3 block1(16, 16);
     *   dim3 grid1( (D_H  + 15) / 16,
     *               (N    + 15) / 16 );
     *   matmul_relu_kernel<<<grid1, block1, 0, stream>>>(
     *       d_X, d_W1, d_b1, d_H, N, D_IN, D_H);
     *   CUDA_CHECK( cudaGetLastError() );
     */

    /* ── Kernel 2: Output layer ──────────────────────────────────────────────── */
    /*
     * Same idea but the output is Y (N × D_OUT).
     */

    /* TODO 17 — Set up dim3 block and dim3 grid, then launch matmul_kernel
     *           into your stream.
     *
     *   dim3 block2(16, 16);
     *   dim3 grid2( (D_OUT + 15) / 16,
     *               (N    + 15) / 16 );
     *   matmul_kernel<<<grid2, block2, 0, stream>>>(
     *       d_H, d_W2, d_b2, d_Y, N, D_H, D_OUT);
     *   CUDA_CHECK( cudaGetLastError() );
     */

    /* ── Async Device → Host copy ────────────────────────────────────────────── */

    /* TODO 18 — Copy Y back to host asynchronously in the same stream.
     *
     *   CUDA_CHECK( cudaMemcpyAsync(h_Y, d_Y,
     *                               N * D_OUT * sizeof(float),
     *                               cudaMemcpyDeviceToHost, stream) );
     */

    /* ── Synchronise & clean up ──────────────────────────────────────────────── */

    /* TODO 19 — Wait for all stream operations to finish, then destroy the stream.
     *
     *   CUDA_CHECK( cudaStreamSynchronize(stream) );
     *   CUDA_CHECK( cudaStreamDestroy(stream) );
     */
}


/* ═════════════════════════════════════════════════════════════════════════════
 * MAIN — do not modify
 * ═════════════════════════════════════════════════════════════════════════════*/
int main(void)
{
    /* ── Sizes ───────────────────────────────────────────────────────────────── */
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
    CUDA_CHECK( cudaMallocHost(&h_H_cpu, sz_H)  );   /* CPU hidden activations */
    CUDA_CHECK( cudaMallocHost(&h_Y_cpu, sz_Y)  );   /* CPU output             */
    CUDA_CHECK( cudaMallocHost(&h_Y_gpu, sz_Y)  );   /* GPU output             */

    /* ── Initialise with deterministic pseudo-random values ──────────────────── */
    srand(42);
    for (int i = 0; i < N * D_IN;   ++i) h_X [i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < D_IN * D_H;  ++i) h_W1[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < D_H;          ++i) h_b1[i] = (rand() % 40  -  20) / 100.0f;
    for (int i = 0; i < D_H * D_OUT;  ++i) h_W2[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < D_OUT;         ++i) h_b2[i] = (rand() % 40  -  20) / 100.0f;

    /* ── CPU reference ───────────────────────────────────────────────────────── */
    mlp_cpu(h_X, h_W1, h_b1, h_W2, h_b2, h_H_cpu, h_Y_cpu);
    printf("[CPU] MLP forward pass complete.\n");

    /* ── Device allocations ──────────────────────────────────────────────────── */
    float *d_X, *d_W1, *d_b1, *d_W2, *d_b2, *d_H, *d_Y;
    CUDA_CHECK( cudaMalloc(&d_X,  sz_X)  );
    CUDA_CHECK( cudaMalloc(&d_W1, sz_W1) );
    CUDA_CHECK( cudaMalloc(&d_b1, sz_b1) );
    CUDA_CHECK( cudaMalloc(&d_W2, sz_W2) );
    CUDA_CHECK( cudaMalloc(&d_b2, sz_b2) );
    CUDA_CHECK( cudaMalloc(&d_H,  sz_H)  );
    CUDA_CHECK( cudaMalloc(&d_Y,  sz_Y)  );

    /* ── GPU forward pass ────────────────────────────────────────────────────── */
    mlp_forward_gpu(h_X,  h_W1, h_b1, h_W2, h_b2,
                    h_Y_gpu,
                    d_X,  d_W1, d_b1, d_W2, d_b2, d_H, d_Y);
    printf("[GPU] MLP forward pass complete.\n");

    /* ── Correctness check ───────────────────────────────────────────────────── */
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

    /* ── Cleanup ─────────────────────────────────────────────────────────────── */
    cudaFree(d_X);  cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_H);  cudaFree(d_Y);
    /* Pinned memory must be released with cudaFreeHost, not free() */
    cudaFreeHost(h_X);  cudaFreeHost(h_W1); cudaFreeHost(h_b1);
    cudaFreeHost(h_W2); cudaFreeHost(h_b2);
    cudaFreeHost(h_H_cpu); cudaFreeHost(h_Y_cpu); cudaFreeHost(h_Y_gpu);
    return 0;
}
