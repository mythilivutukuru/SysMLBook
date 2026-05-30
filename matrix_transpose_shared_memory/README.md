# Matrix Transpose using Shared Memory

Matrix transpose is a fundamental linear algebra operation widely used in scientific computing and deep learning workloads. A naive GPU transpose suffers from uncoalesced memory accesses, either on the read or write side. This task requires implementing an efficient CUDA matrix transpose that exploits **shared memory tiling** to ensure coalesced global memory access patterns.

## Background

Given a square input matrix $A \in \mathbb{R}^{m \times m}$, the transpose $C = A^T$ satisfies:

$$C[j, i] = A[i, j] \quad \forall\; 0 \leq i, j < m$$

Matrices are stored in **row-major flattened** format. The macro:

```c
#define INDX(row, col, d) (((row) * (d)) + (col))
```

is used throughout to index a 1D array with 2D coordinates, where `d` is the number of columns. This macro must be used in the kernel.

The key idea behind the shared memory transpose is to decompose the matrix into **tiles** of size 32×32. Each thread block:
1. Loads one tile from global memory into shared memory in a coalesced fashion
2. Synchronizes
3. Writes the transposed tile to global memory, also in a coalesced fashion

This avoids the uncoalesced writes that arise in a naive transpose.

### Example

```
Input A:                    Output A^T:
 1   2   3   4              1   5   9  13
 5   6   7   8     --->     2   6  10  14
 9  10  11  12              3   7  11  15
13  14  15  16              4   8  12  16
```

Each thread block contains 32×32 threads and processes exactly one 32×32 tile of the matrix. For matrices whose dimensions are not multiples of 32, boundary threads must be guarded so they perform no memory access outside the valid matrix range.

## Task

Implement two functions inside `student.cu`. **Do not modify any other code.**

---

### Kernel: `smem_cuda_transpose`

```c
__global__ void smem_cuda_transpose(int m, float* a, float* c)
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `m` | Side length of the square matrix |
| `a` | Input matrix in row-major flattened format (size: m × m) |
| `c` | Output (transposed) matrix in row-major flattened format (size: m × m) |

**Requirements:**
- Declare a `__shared__` array of size 32×32 for the tile
- Each thread reads one element from `a` into shared memory using `INDX`
- Call `__syncthreads()` after the shared memory load
- Each thread writes the transposed element from shared memory to `c` using `INDX`
- Use **separate boundary guards** for the read and write steps — when `m` is not a multiple of 32, edge tiles are only partially filled, and the valid thread ranges differ between the two steps. Guard each step independently using appropriate row and column bounds.

---

### Launch Wrapper: `gpu_transpose`

```c
void gpu_transpose(int m, float* h_a, float* h_c)
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `m` | Side length of the square matrix |
| `h_a` | Host pointer to the input matrix |
| `h_c` | Host pointer to the output matrix (to be filled) |

**Requirements:**
- Allocate device memory for the input and output matrices using `cudaMalloc`
- Copy `h_a` to the device using `cudaMemcpy`
- Configure a 2D grid with thread blocks of size 32×32; use ⌈m/32⌉ blocks along each dimension
- Launch `smem_cuda_transpose`
- Synchronize with `cudaDeviceSynchronize`
- Copy the result back to `h_c` using `cudaMemcpy`
- Free all device memory with `cudaFree`

## Testing

Graded on the provided test cases in `main()`. Each test case calls `gpu_transpose` and compares it element-wise against a CPU reference transpose. A test case passes only if **every element matches** within a tolerance of ε = 10⁻⁴

## Notes

- Only modify the two functions described above; all other code is fixed
- Matrices are stored in row-major flattened format; always use the `INDX` macro for indexing
- Matrix dimensions are not necessarily multiples of 32 or powers of 2 — **boundary handling is essential**
- Tile size is fixed at 32×32
