#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

const int MAX_ELEMS_PER_THREAD = 8;

__global__ void matmul_tiled_row_kernel(const float* A, const float* B, float* C,
                                              int N, int TILE, int elems_per_thread) {
    extern __shared__ float s[]; // size must be (TILE*TILE*(1 + elems_per_thread)) floats
    float* sA = s; // TILE * TILE
    float* sB = s + TILE * TILE; // TILE * (TILE * elems_per_thread)

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int localRow = threadIdx.y; // in [0, TILE)
    int localCol = threadIdx.x; // in [0, TILE)

    // base global row/col for this thread-block
    int globalRow = blockRow * TILE + localRow;
    int baseGlobalCol = blockCol * (TILE * elems_per_thread); // block covers TILE * elems_per_thread columns

    float acc[MAX_ELEMS_PER_THREAD];
    for (int e = 0; e < elems_per_thread; ++e) acc[e] = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;

    // stride per row in sB (number of columns in sB)
    int sB_row_stride = TILE * elems_per_thread;

    for (int t = 0; t < numTiles; ++t) {
        // --- load A tile element ---
        int aCol = t * TILE + localCol; // column inside A
        if (globalRow < N && aCol < N) {
            sA[localRow * TILE + localCol] = A[globalRow * N + aCol];
        } else {
            sA[localRow * TILE + localCol] = 0.0f;
        }

        // --- load B tile: load all elems_per_thread sub-tiles for this thread ---
        int bRow = t * TILE + localRow; // row in B that corresponds to this tile's row
        for (int e = 0; e < elems_per_thread; ++e) {
            // global column for this sub-tile element
            int gCol = baseGlobalCol + localCol + e * TILE;
            // Write into sB at row localRow, column (localCol + e*TILE)
            if (bRow < N && gCol < N) {
                sB[ localRow * sB_row_stride + (localCol + e * TILE) ] = B[bRow * N + gCol];
            } else {
                sB[ localRow * sB_row_stride + (localCol + e * TILE) ] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            float aval = sA[ localRow * TILE + k ];
            for (int e = 0; e < elems_per_thread; ++e) {
                int colOffset = localCol + e * TILE; 
                float bval = sB[ k * sB_row_stride + colOffset ];
                acc[e] += aval * bval;
            }
        }

        __syncthreads();
    } 

    // write results back to global C
    for (int e = 0; e < elems_per_thread; ++e) {
        int gcol = baseGlobalCol + localCol + e * TILE;
        if (globalRow < N && gcol < N) {
            C[ globalRow * N + gcol ] = acc[e];
        }
    }
}


void matmul_gpu_row(int N, int TILE, int elems_per_thread, float* A_h, float* B_h, float* C_h, float &elapsed_ms) {
    float *A_d, *B_d, *C_d;
    size_t bytes = (size_t)N * N * sizeof(float);
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    // block: TILE x TILE threads
    dim3 block(TILE, TILE);

    // grid.x must account for elems_per_thread (each block covers TILE * elems_per_thread columns)
    int blocks_x = (N + TILE * elems_per_thread - 1) / (TILE * elems_per_thread);
    int blocks_y = (N + TILE - 1) / TILE;
    dim3 grid(blocks_x, blocks_y);

    // shared memory: TILE*TILE for sA + TILE * (TILE*elems) for sB
    size_t sharedFloats = (size_t)TILE * TILE * (1 + elems_per_thread);
    size_t sharedBytes = sharedFloats * sizeof(float);

    cudaEvent_t s, e; 
    cudaEventCreate(&s); 
    cudaEventCreate(&e); 
    cudaEventRecord(s);
    matmul_tiled_row_kernel<<<grid, block, sharedBytes>>>(A_d, B_d, C_d, N, TILE, elems_per_thread);
    cudaEventRecord(e); 
    cudaEventSynchronize(e); 
    cudaEventElapsedTime(&elapsed_ms, s, e);

    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(A_d); 
    cudaFree(B_d); 
    cudaFree(C_d); 
    cudaEventDestroy(s); 
    cudaEventDestroy(e);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./matmul_tiled_row <N> <TILE> <elems_per_thread>\n"; 
        return 1; 
    }
    int N = atoi(argv[1]); 
    int TILE = atoi(argv[2]); 
    int elems = atoi(argv[3]);
    std::cout << "Row-per-thread TILED matmul N=" << N << " TILE=" << TILE << " elems_per_thread=" << elems << "\n";

    size_t bytes = (size_t)N * N * sizeof(float);
    float* A = (float*)malloc(bytes); float* B = (float*)malloc(bytes); float* C = (float*)malloc(bytes);
    for (size_t i = 0; i < (size_t)N * N; ++i) { A[i] = float(i % 100); B[i] = float((i * 7) % 100); }

    float ms; matmul_gpu_row(N, TILE, elems, A, B, C, ms);
    std::cout << "Kernel time (ms): " << ms << "\n";
    free(A); free(B); free(C); return 0;
}