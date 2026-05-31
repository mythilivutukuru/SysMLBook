# CUDA Matrix Multiplication Assignment

In this assignment, you will implement matrix multiplication two ways:

1. **A custom CUDA kernel** — write the GPU kernel yourself, managing thread indexing and global memory access.
2. **cuBLAS** — call NVIDIA's routine and understand how to adapt row-major C arrays to cuBLAS's column-major convention.

You will verify both implementations against a CPU reference.

---

## Background

### CUDA Programming Model

A CUDA kernel is launched with a **grid** of **thread blocks**. Each block contains up to 1024 threads, arranged in up to three dimensions. Threads within a block can synchronise and share fast **shared memory**; threads across blocks cannot.

```c
// Declare a kernel
__global__ void my_kernel(float *A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] *= 2.0f;
}

// Launch it: 256 blocks of 128 threads each
my_kernel<<<256, 128>>>(d_ptr, n);
```

For 2-D problems (like matrices) use `dim3` for both the grid and the block:

```c
dim3 block(16, 16);                                    // 256 threads/block
dim3 grid((N+15)/16, (N+15)/16);                       // enough blocks to cover N×N
kernel<<<grid, block>>>(d_A, d_B, d_C, N);
```

Inside the kernel, thread coordinates are:

```c
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

A **row-major** matrix element `M[row][col]` is stored at `M[row * n + col]`.

### cuBLAS — `cublasSgemm`

cuBLAS is NVIDIA's GPU-accelerated BLAS library. Its single-precision GEMM routine computes **C = α·op(A)·op(B) + β·C**:

```c
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,   // CUBLAS_OP_N = no transpose
    cublasOperation_t transb,
    int m,                      // rows of op(A) and C
    int n,                      // cols of op(B) and C
    int k,                      // cols of op(A) = rows of op(B)
    const float *alpha,
    const float *A, int lda,    // A and its leading dimension
    const float *B, int ldb,    // B and its leading dimension
    const float *beta,
    float *C, int ldc           // output C and its leading dimension
);
```

**Column-major convention:** cuBLAS stores matrices in **column-major** order (Fortran style). C arrays are **row-major**. The standard workaround uses the identity:

> C = A · B  ⟺  Cᵀ = Bᵀ · Aᵀ

Because cuBLAS sees our row-major `A` as a column-major `Aᵀ` (and similarly for B and C), we can compute the correct row-major result by swapping the operand order and swapping `m` ↔ `n`:

```c
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n,      // m  (cols of Bᵀ)
            n,      // n  (rows of Aᵀ)
            n,      // k  (inner dim)
            &alpha,
            d_B, n, // pass B first
            d_A, n, // then A
            &beta,
            d_C, n);
```

**cuBLAS handle lifecycle:**

```c
cublasHandle_t handle;
cublasCreate(&handle);
// ... use handle ...
cublasDestroy(handle);
```

---

## Task

Open `student.cu`. There are **two functions to implement**:

### Task 1 — `matmul_gpu` (CUDA kernel)

Write a `__global__` kernel where **each thread computes one element** of the output matrix C.

- Map thread indices `(threadIdx, blockIdx, blockDim)` to a `(row, col)` in C.
- Guard against out-of-bounds threads (when N is not a multiple of block size).
- Compute the dot product of row `row` of A with column `col` of B.
- Write the result to `C[row * n + col]`.

### Task 2 — `matmul_cublas`

Call `cublasSgemm` to compute C = A × B for row-major matrices.

- A `cublasHandle_t` is passed in — do not create or destroy it here.
- Use the column-major workaround described above (swap operands, swap m/n).
- Set `alpha = 1.0` and `beta = 0.0` (plain multiply, no accumulation).

---

## Testing

### Build

```bash
nvcc -o matmul student.cu -lcublas
./matmul
```

Expected output (timings will vary by GPU):

```
Matrix size: 1024 x 1024

Running CPU reference (this may take a while)...
[CPU ref]      Time: 4821.5 ms

[GPU kernel]   Time:  12.34 ms  | CORRECT
[cuBLAS]       Time:   0.87 ms  | CORRECT
```

### Debugging tips

- Start with a small matrix (e.g. change `N` to `64`) and print a few values.
- If you see `WRONG`, the mismatch location is printed — check your index arithmetic around that `[row, col]`.
- For cuBLAS, double-check the operand order (`d_B` before `d_A`) and that `m`, `n`, `k` are all set to `N`.