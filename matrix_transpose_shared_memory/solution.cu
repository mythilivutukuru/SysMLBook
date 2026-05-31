#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

// Row-major indexing macro
#define INDX(row, col, d) (((row) * (d)) + (col))

/**
 * @brief CUDA kernel to transpose a square matrix using shared memory tiling.
 *
 * Each block loads a tile into shared memory, synchronizes,
 * then writes the transposed tile back to global memory.
 *
 * @param m   Size of the matrix (m x m)
 * @param a   Input matrix in device memory
 * @param c   Output (transposed) matrix in device memory
 */
__global__ void smem_cuda_transpose(int m, float *a, float *c)
{
    // Shared memory tile
    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

    // Global indices
    const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

    // Tile origin indices
    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    // -------------------------
    // READ from global memory
    // -------------------------
    if (myRow < m && myCol < m)
    {
        smemArray[threadIdx.x][threadIdx.y] =
            a[INDX(myRow, myCol, m)];
    }

    // Ensure all threads finish loading tile
    __syncthreads();

    // -------------------------
    // WRITE transposed tile
    // -------------------------
    if ((tileY + threadIdx.x) < m &&
        (tileX + threadIdx.y) < m)
    {
        c[INDX(tileY + threadIdx.x,
               tileX + threadIdx.y, m)] =
            smemArray[threadIdx.y][threadIdx.x];
    }
}

/**
 * @brief Host function to allocate device memory and launch transpose kernel.
 *
 * @param m     Size of matrix (m x m)
 * @param h_a   Host input matrix
 * @param h_c   Host output matrix
 */
void gpu_transpose(int m, float *h_a, float *h_c)
{
    size_t bytes = (size_t)m * m * sizeof(float);

    float *d_a, *d_c;

    // Allocate device memory
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // Copy input matrix to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    // Initialize output matrix
    cudaMemset(d_c, 0, bytes);

    // Configure execution geometry (number of threads and blocks)
    dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 blocks(
        (m + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X,
        (m + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y
    );

    // Launch kernel
    smem_cuda_transpose<<<blocks, threads>>>(m, d_a, d_c);

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_c);
}

/* ============================================================
   TESTING INFRASTRUCTURE — students do not modify below
   ============================================================ */

/**
 * @brief CPU reference transpose for validation.
 *
 * @param m   Matrix size
 * @param a   Input matrix
 * @param c   Output transposed matrix
 */
void cpu_transpose(int m, float *a, float *c)
{
    for (int col = 0; col < m; col++)
        for (int row = 0; row < m; row++)
            c[INDX(col, row, m)] =
                a[INDX(row, col, m)];
}

/**
 * @brief Fill matrix with random float values.
 *
 * @param m   Matrix size
 * @param a   Matrix to fill
 */
void fill_random(int m, float *a)
{
    for (int i = 0; i < m * m; i++)
        a[i] = (float)(rand() % 100) / 10.0f;
}

/**
 * @brief Compare two matrices element-wise.
 *
 * @param m     Matrix size
 * @param ref   Reference matrix
 * @param test  Test matrix
 * @param tol   Allowed tolerance
 * @return 1 if equal within tolerance, otherwise 0
 */
int compare_matrices(int m, float *ref, float *test, float tol)
{
    for (int i = 0; i < m * m; i++)
    {
        if (fabs(ref[i] - test[i]) > tol)
        {
            printf(" MISMATCH at index %d: ref=%.4f, gpu=%.4f\n",
                   i, ref[i], test[i]);
            return 0;
        }
    }
    return 1;
}

/**
 * @brief Run a single test case.
 *
 * @param m Matrix size
 */
void run_test(int m)
{
    printf("=== Test: %d x %d matrix ===\n", m, m);

    size_t bytes = (size_t)m * m * sizeof(float);

    float *h_a      = (float*)malloc(bytes);
    float *h_c_cpu  = (float*)malloc(bytes);
    float *h_c_gpu  = (float*)malloc(bytes);

    fill_random(m, h_a);

    cpu_transpose(m, h_a, h_c_cpu);
    gpu_transpose(m, h_a, h_c_gpu);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf(" CUDA error: %s\n",
               cudaGetErrorString(err));

    int pass = compare_matrices(m, h_c_cpu, h_c_gpu, 1e-4f);

    printf(" Result: %s\n\n",
           pass ? "PASS" : "FAIL");

    free(h_a);
    free(h_c_cpu);
    free(h_c_gpu);
}

/**
 * @brief Main.
 */
int main(void)
{
    srand(42);

    run_test(32);   // Small
    run_test(128);  // Medium
    run_test(513);  // Non-multiple of tile size

    return 0;
}