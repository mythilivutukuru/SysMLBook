#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ── dimensions ────────────────────────────────────────────────────────────── */
#define BATCH   64
#define IN_DIM  128
#define HIDDEN  256
#define OUT_DIM 10

/* ── error-checking helper ─────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 1 — Matrix Multiplication with bias
 *
 * Computes:  C[row][col] = sum_k( A[row][k] * B[k][col] ) + bias[col]
 *
 * Arguments:
 *   A    — input  matrix, shape (M x K), row-major
 *   B    — weight matrix, shape (K x N), row-major
 *   bias — bias vector,   length N  (one value per output column)
 *   C    — output matrix, shape (M x N), row-major
 *   M, K, N — matrix dimensions
 *
 * Launch configuration (set in mlp_forward below):
 *   grid  — 2-D grid of blocks covering the (M x N) output
 *   block — 2-D block of threads (e.g. 16×16 or 32×32)
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void matmul_kernel(const float *A, const float *B,
                               const float *bias, float *C,
                               int M, int K, int N)
{
    /* TODO 1 ────────────────────────────────────────────────────────────────
     *
     * Step 1: Compute the (row, col) index this thread is responsible for.
     *         Use blockIdx, blockDim, and threadIdx.
     *         - row = which row   of C does this thread compute?
     *         - col = which column of C does this thread compute?
     *
     * Step 2: Bounds-check.
     *         Return early if row >= M  or  col >= N.
     *         (The grid may be slightly larger than the matrix.)
     *
     * Step 3: Accumulate the dot product.
     *         Loop k from 0 to K-1:
     *             sum += A[row * K + k] * B[k * N + col]
     *
     * Step 4: Write the result with bias.
     *         C[row * N + col] = sum + bias[col]
     *
     * ──────────────────────────────────────────────────────────────────── */
}


/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 2 — ReLU activation (in-place)
 *
 * Applies ReLU element-wise: M[i] = max(0, M[i])
 *
 * Arguments:
 *   M       — matrix stored as a flat 1-D array of (rows * cols) floats
 *   rows    — number of rows
 *   cols    — number of columns
 *
 * Launch configuration (set in mlp_forward below):
 *   Each thread handles ONE element.
 *   You can use a 1-D or 2-D grid — either works.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void relu_kernel(float *M, int rows, int cols)
{
    /* TODO 2 ────────────────────────────────────────────────────────────────
     *
     * Step 1: Compute the global index of the element this thread owns.
     *         For a 1-D launch:  idx = blockIdx.x * blockDim.x + threadIdx.x
     *         For a 2-D launch:  compute (row, col) then idx = row*cols + col
     *
     * Step 2: Bounds-check.
     *         Return early if idx >= rows * cols.
     *
     * Step 3: Apply ReLU.
     *         M[idx] = (M[idx] > 0.f) ? M[idx] : 0.f;
     *         (or equivalently: M[idx] = fmaxf(0.f, M[idx]);)
     *
     * ──────────────────────────────────────────────────────────────────── */
}


/* ═══════════════════════════════════════════════════════════════════════════
 * LAUNCH WRAPPER — mlp_forward
 *
 * Runs the full two-layer forward pass on the GPU:
 *
 *   Z1  = X  @ W1 + b1    (matmul_kernel)
 *   A1  = ReLU(Z1)         (relu_kernel)
 *   Z2  = A1 @ W2 + b2    (matmul_kernel)
 *   out = Z2
 *
 * All pointers are already on the DEVICE (allocated by the caller).
 *
 * Arguments:
 *   d_X   (BATCH  x IN_DIM)  — input batch
 *   d_W1  (IN_DIM x HIDDEN)  — layer-1 weights
 *   d_b1  (HIDDEN)           — layer-1 biases
 *   d_W2  (HIDDEN  x OUT_DIM)— layer-2 weights
 *   d_b2  (OUT_DIM)          — layer-2 biases
 *   d_out (BATCH  x OUT_DIM) — final output (pre-allocated by caller)
 * ═══════════════════════════════════════════════════════════════════════════ */
void mlp_forward(const float *d_X,
                 const float *d_W1, const float *d_b1,
                 const float *d_W2, const float *d_b2,
                 float       *d_out)
{
    /* TODO 3 ────────────────────────────────────────────────────────────────
     *
     * Step 1: Allocate an intermediate device buffer for Z1 / A1.
     *         Size: BATCH * HIDDEN * sizeof(float)
     *         Use cudaMalloc.
     *
     * Step 2: Choose block and grid dimensions for the matmul launches.
     *         Suggestion — 2-D block of 16×16 threads:
     *             dim3 block(16, 16);
     *         Grid must cover the full output matrix:
     *             dim3 grid( (N + 15)/16, (M + 15)/16 );
     *         (swap x/y carefully: x covers columns, y covers rows)
     *
     * Step 3: Launch matmul_kernel for Z1 = X @ W1 + b1.
     *         M=BATCH, K=IN_DIM, N=HIDDEN
     *
     * Step 4: Launch relu_kernel on Z1 (in-place → result is A1).
     *         Total elements: BATCH * HIDDEN
     *         Suggestion — 1-D block of 256 threads:
     *             int threads = 256;
     *             int blocks  = (BATCH * HIDDEN + threads - 1) / threads;
     *
     * Step 5: Launch matmul_kernel for Z2 = A1 @ W2 + b2.
     *         M=BATCH, K=HIDDEN, N=OUT_DIM
     *         Output goes into d_out.
     *
     * Step 6: Synchronize — cudaDeviceSynchronize()
     *         Check errors with CUDA_CHECK.
     *
     * Step 7: Free the intermediate buffer (cudaFree).
     *
     * ──────────────────────────────────────────────────────────────────── */
}


/* ═══════════════════════════════════════════════════════════════════════════
 * CPU REFERENCE IMPLEMENTATION
 * Used to verify your CUDA output.  Do NOT modify.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void cpu_matmul_bias(const float *A, const float *B, const float *bias,
                             float *C, int M, int K, int N)
{
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++)
                sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = sum + bias[col];
        }
    }
}

static void cpu_relu(float *M, int rows, int cols)
{
    int total = rows * cols;
    for (int i = 0; i < total; i++)
        M[i] = (M[i] > 0.f) ? M[i] : 0.f;
}

static void cpu_mlp_forward(const float *X,
                             const float *W1, const float *b1,
                             const float *W2, const float *b2,
                             float *out)
{
    /* Allocate intermediate buffer on the CPU heap */
    float *Z1 = (float *)malloc(BATCH * HIDDEN * sizeof(float));

    cpu_matmul_bias(X,  W1, b1, Z1,  BATCH, IN_DIM, HIDDEN);
    cpu_relu(Z1, BATCH, HIDDEN);
    cpu_matmul_bias(Z1, W2, b2, out, BATCH, HIDDEN, OUT_DIM);

    free(Z1);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN — data setup, kernel launch, correctness check
 * Do NOT modify.
 * ═══════════════════════════════════════════════════════════════════════════ */
static float randf() { return (float)rand() / RAND_MAX * 2.f - 1.f; }

int main(void)
{
    srand(42);
    printf("Running 2-layer MLP forward pass...\n");
    printf("  Batch=%-4d  in=%-4d  hidden=%-4d  out=%d\n",
           BATCH, IN_DIM, HIDDEN, OUT_DIM);

    /* ── allocate and fill host arrays ──────────────────────────────────── */
    size_t sz_X  = BATCH   * IN_DIM  * sizeof(float);
    size_t sz_W1 = IN_DIM  * HIDDEN  * sizeof(float);
    size_t sz_b1 = HIDDEN             * sizeof(float);
    size_t sz_W2 = HIDDEN  * OUT_DIM * sizeof(float);
    size_t sz_b2 = OUT_DIM            * sizeof(float);
    size_t sz_O  = BATCH   * OUT_DIM * sizeof(float);

    float *h_X   = (float *)malloc(sz_X);
    float *h_W1  = (float *)malloc(sz_W1);
    float *h_b1  = (float *)malloc(sz_b1);
    float *h_W2  = (float *)malloc(sz_W2);
    float *h_b2  = (float *)malloc(sz_b2);
    float *h_out_cpu  = (float *)malloc(sz_O);
    float *h_out_cuda = (float *)malloc(sz_O);

    for (int i = 0; i < BATCH  * IN_DIM;  i++) h_X [i] = randf();
    for (int i = 0; i < IN_DIM * HIDDEN;  i++) h_W1[i] = randf();
    for (int i = 0; i < HIDDEN;            i++) h_b1[i] = randf();
    for (int i = 0; i < HIDDEN * OUT_DIM; i++) h_W2[i] = randf();
    for (int i = 0; i < OUT_DIM;           i++) h_b2[i] = randf();

    /* ── CPU reference ───────────────────────────────────────────────────── */
    cpu_mlp_forward(h_X, h_W1, h_b1, h_W2, h_b2, h_out_cpu);
    printf("[CPU]  reference done.\n");

    /* ── allocate device memory and copy inputs ──────────────────────────── */
    float *d_X, *d_W1, *d_b1, *d_W2, *d_b2, *d_out;
    CUDA_CHECK(cudaMalloc(&d_X,   sz_X));
    CUDA_CHECK(cudaMalloc(&d_W1,  sz_W1));
    CUDA_CHECK(cudaMalloc(&d_b1,  sz_b1));
    CUDA_CHECK(cudaMalloc(&d_W2,  sz_W2));
    CUDA_CHECK(cudaMalloc(&d_b2,  sz_b2));
    CUDA_CHECK(cudaMalloc(&d_out, sz_O));

    CUDA_CHECK(cudaMemcpy(d_X,  h_X,  sz_X,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1, sz_W1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, sz_b1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2, sz_W2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, sz_b2, cudaMemcpyHostToDevice));

    /* ── run CUDA forward pass ───────────────────────────────────────────── */
    mlp_forward(d_X, d_W1, d_b1, d_W2, d_b2, d_out);
    printf("[CUDA] kernel done.\n");

    /* ── copy result back ────────────────────────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(h_out_cuda, d_out, sz_O, cudaMemcpyDeviceToHost));

    /* ── correctness check ───────────────────────────────────────────────── */
    float max_err = 0.f;
    int   err_idx = -1;
    int   total   = BATCH * OUT_DIM;
    for (int i = 0; i < total; i++) {
        float diff = fabsf(h_out_cuda[i] - h_out_cpu[i]);
        if (diff > max_err) { max_err = diff; err_idx = i; }
    }
    printf("Max absolute error: %e\n", max_err);
    if (max_err < 1e-3f) {
        printf("Result: PASSED\n");
    } else {
        printf("Result: FAILED  (first bad index=%d  cuda=%.6f  cpu=%.6f)\n",
               err_idx, h_out_cuda[err_idx], h_out_cpu[err_idx]);
    }

    /* ── cleanup ─────────────────────────────────────────────────────────── */
    cudaFree(d_X); cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_out);
    free(h_X); free(h_W1); free(h_b1);
    free(h_W2); free(h_b2); free(h_out_cpu); free(h_out_cuda);

    return 0;
}
