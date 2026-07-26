#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

const int MAX_ELEMS_PER_THREAD = 8;

__global__ void matmul_tiled_registers_kernel(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int N, int TILE, int elems_per_thread) {
    extern __shared__ float s[]; // size = TILE * (TILE * elems_per_thread) floats
    float* sB = s;

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int localRow = threadIdx.y;   // [0 .. TILE-1]
    int localCol = threadIdx.x;   // [0 .. TILE-1]

    int globalRow = blockRow * TILE + localRow;
    int baseGlobalCol = blockCol * (TILE * elems_per_thread); // block covers TILE*elems columns

    // accumulators
    float acc[MAX_ELEMS_PER_THREAD];
    for (int e = 0; e < elems_per_thread; ++e) acc[e] = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;
    int sB_row_stride = TILE * elems_per_thread; // number of columns in sB per row

    for (int t = 0; t < numTiles; ++t) {
        int bRow = t * TILE + localRow;
        for (int e = 0; e < elems_per_thread; ++e) {
            int gCol = baseGlobalCol + localCol + e * TILE;
            float val = 0.0f;
            if (bRow < N && gCol < N) val = B[bRow * N + gCol];
            sB[ localRow * sB_row_stride + (localCol + e * TILE) ] = val;
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            // load A element into register
            int aCol = t * TILE + k;
            float aval = 0.0f;
            if (globalRow < N && aCol < N) aval = A[globalRow * N + aCol];

            // multiply-accumulate for each output column this thread is responsible for
            // colOffset within sB row is (localCol + e*TILE)
            for (int e = 0; e < elems_per_thread; ++e) {
                int colOffset = localCol + e * TILE;
                // bval from shared memory
                float bval = sB[ k * sB_row_stride + colOffset ];
                acc[e] += aval * bval;
            }
        }

        __syncthreads();
    }

    // write results back
    for (int e = 0; e < elems_per_thread; ++e) {
        int gcol = baseGlobalCol + localCol + e * TILE;
        if (globalRow < N && gcol < N) {
            C[globalRow * N + gcol] = acc[e];
        }
    }
}

void matmul_gpu_registers(int N, int TILE, int elems_per_thread,
                          float* A_h, float* B_h, float* C_h, float &elapsed_ms) {
    if (elems_per_thread <= 0 || elems_per_thread > MAX_ELEMS_PER_THREAD) {
        std::cerr << "elems_per_thread must be in 1.." << MAX_ELEMS_PER_THREAD << "\n";
        exit(1);
    }
    if (TILE <= 0 || TILE * TILE > 1024) {
        std::cerr << "Invalid TILE: TILE>0 and TILE*TILE <= 1024 required\n";
        exit(1);
    }

    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    size_t bytes = (size_t)N * N * sizeof(float);
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    int blocks_x = (N + TILE * elems_per_thread - 1) / (TILE * elems_per_thread);
    int blocks_y = (N + TILE - 1) / TILE;
    dim3 grid(blocks_x, blocks_y);

    size_t sharedFloats = (size_t)TILE * (TILE * elems_per_thread);
    size_t sharedBytes = sharedFloats * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_tiled_registers_kernel<<<grid, block, sharedBytes>>>(A_d, B_d, C_d, N, TILE, elems_per_thread);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc,char** argv){ 
    if(argc<4){
        std::cerr<<"Usage: ./matmul_tiled_registers <N> <TILE> <elems_per_thread>\n"; 
        return 1;
    } 
    int N=atoi(argv[1]); 
    int TILE=atoi(argv[2]); 
    int elems=atoi(argv[3]); 
    std::cout<<"Register-block TILED matmul N="<<N<<" TILE="<<TILE<<" elems="<<elems<<"\n";
    size_t bytes=(size_t)N*N*sizeof(float); 
    float *A=(float*)malloc(bytes),*B=(float*)malloc(bytes),*C=(float*)malloc(bytes); 
    for(size_t i=0;i<(size_t)N*N;++i){
        A[i]=float(i%100);
        B[i]=float((i*7)%100);
    } 
    float ms; 
    matmul_gpu_registers(N,TILE,elems,A,B,C,ms); 
    std::cout<<"Kernel time (ms): "<<ms<<"\n"; 
    free(A);
    free(B);
    free(C); 
    return 0; 
}