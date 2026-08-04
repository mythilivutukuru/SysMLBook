#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>

/**
 * @brief Tiled matrix multiplication kernel using 2D block tiling
 *        and per-thread register blocking.
 *
 * Each thread computes an R x R sub-block of C in registers.
 * Shared memory stores one TILE x TILE tile of A and B.
 *
 * @tparam R Register tile size (per-thread output tile dimension)
 *
 * @param A Pointer to matrix A (device memory)
 * @param B Pointer to matrix B (device memory)
 * @param C Pointer to matrix C (device memory)
 * @param N Matrix dimension (NxN)
 * @param TILE Tile size loaded into shared memory
 */
template<int R>
__global__
void matmul_tiled_block2d_kernel(const float* A,
                                 const float* B,
                                 float* C,
                                 int N,
                                 int TILE)
{
    // Shared memory layout:
    // [ sA (TILE x TILE) | sB (TILE x TILE) ]
    extern __shared__ float s[];
    float* sA = s;
    float* sB = s + TILE*TILE;

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /// Compute global base row/column for this thread's R x R block
    int rowBase = (blockIdx.y * TILE) + ty * R;
    int colBase = (blockIdx.x * TILE) + tx * R;

    /// Register accumulation tile
    float reg[R][R];

    for (int i = 0; i < R; i++)
        for (int j = 0; j < R; j++)
            reg[i][j] = 0.0f;

    // Number of tiles along K dimension
    int numTiles = (N + TILE - 1) / TILE;


    for (int t = 0; t < numTiles; t++) {
        // TODO: Load tile of A into shared memory

        // TODO: Load tile of B into shared memory

        // TODO: Multiply shared memory tiles

    }

    
    // TODO: Write register tile back to global memory

}

/**
 * @brief Launch GPU block-tiling matrix multiplication.
 *
 * @tparam R Register tile size per thread.
 *
 * @param N Matrix dimension (NxN)
 * @param TILE Shared-memory tile dimension
 * @param A_h Host pointer to matrix A
 * @param B_h Host pointer to matrix B
 * @param C_h Host pointer to matrix C
 */
template<int R>
void matmul_gpu_block2d(int N,
                        int TILE,
                        const float* A_h,
                        const float* B_h,
                        float* C_h)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    /// Each thread computes R x R results
    dim3 block(TILE / R, TILE / R);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    size_t sharedBytes = 2 * TILE * TILE * sizeof(float);

    matmul_tiled_block2d_kernel<R>
        <<<grid, block, sharedBytes>>>(A_d, B_d, C_d, N, TILE);

    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


/* ============================================================
   TESTING INFRASTRUCTURE — students do not modify below
   ============================================================ */

/**
 * @brief CPU reference for matrix multiplication.
 *
 * @param N   Matrix size
 * @param A   Input matrix A
 * @param B   Input matrix B
 * @param C   Output matrix
 */
void matmul_cpu(int N,
                const float* A,
                const float* B,
                float* C)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
    }
}

/**
 * @brief Compares two matrices element-wise.
 *
 * @param N   Matrix size
 * @param ref CPU output
 * @param gpu GPU output
 */
bool verify(int N, const float* ref, const float* gpu)
{
    for (int i = 0; i < N*N; i++) {
        if (fabs(ref[i] - gpu[i]) > 1e-3f) {
            std::cout << "Mismatch at " << i
                      << " ref=" << ref[i]
                      << " gpu=" << gpu[i] << "\n";
            return false;
        }
    }
    return true;
}

/**
 * @brief Main.
 */
int main()
{
    constexpr int TILE = 32;
    constexpr int R    = 2;

    std::vector<int> test_sizes = {64, 128, 256};

    for (int N : test_sizes) {

        std::cout << "\n===== Testing N = " << N << " =====\n";

        size_t bytes = (size_t)N * N * sizeof(float);

        std::vector<float> A(N*N), B(N*N), C_gpu(N*N), C_cpu(N*N);

        for (int i = 0; i < N*N; i++) {
            A[i] = float(i % 100);
            B[i] = float((i*7) % 100);
        }

        matmul_cpu(N, A.data(), B.data(), C_cpu.data());

        matmul_gpu_block2d<R>(N, TILE,
                              A.data(),
                              B.data(),
                              C_gpu.data());

        bool ok = verify(N, C_cpu.data(), C_gpu.data());

        std::cout << "Verification: " << (ok ? "PASSED" : "FAILED") << "\n";
    }

    return 0;
}