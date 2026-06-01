# Matrix Multiplication Using Tensor Cores

In this assignment, you will implement a high-performance matrix multiplication kernel in CUDA that leverages **Tensor Cores** via the Warp Matrix Multiply-Accumulate (WMMA) API. You will compute `C = A × B` for large FP16 matrices and compare the result against a CPU reference implementation.

---

## Background

All WMMA functions live in `<mma.h>` under the `nvcuda::wmma` namespace.

### Fragment Declaration

```cpp
wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float>                  acc_frag;
```

| Template Parameter | Meaning |
|--------------------|---------|
| `wmma::matrix_a` / `matrix_b` / `accumulator` | Role of this fragment in the MMA |
| `16, 16, 16` | Tile shape: M=16, N=16, K=16 |
| `half` / `float` | Element type (inputs are FP16, accumulator is FP32) |
| `wmma::row_major` / `col_major` | Memory layout of the source matrix |

### Loading a Fragment

```cpp
wmma::load_matrix_sync(frag, ptr, ldm);
```

- `frag` — the fragment to populate  
- `ptr`  — pointer to the **start of the 16×16 tile** in global memory  
- `ldm`  — leading dimension (stride in elements between consecutive rows)

All 32 threads in the warp must call this together; the hardware distributes the loads automatically.

### Zeroing an Accumulator

```cpp
wmma::fill_fragment(acc_frag, 0.0f);
```

Must be called before the first `mma_sync` that writes to this fragment.

### Performing the MMA

```cpp
wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
```

Computes `acc_frag += a_frag × b_frag` in a single warp-synchronous instruction on the Tensor Core.

### Storing a Fragment

```cpp
wmma::store_matrix_sync(ptr, acc_frag, ldm, wmma::mem_row_major);
```

Writes the 16×16 accumulator tile back to memory at `ptr` with leading dimension `ldm`.

### Hardware Requirements

Tensor Cores (WMMA 16×16×16) require **sm_70 or later** (Volta, Turing, Ampere, …).  

---

## Task

1. **`convertFp32ToFp16`** — convert a row-major FP32 matrix to FP16 on the GPU (helper kernel, skeleton provided).

2. **`matmulTensorCore`** — the main kernel. Each warp is responsible for one 16×16 output tile of `C`. For each K-tile:
   - Load the 16×16 tile of `A` (FP16, row-major) into `a_frag`
   - Load the 16×16 tile of `B` (FP16, col-major) into `b_frag`
   - Accumulate with `mma_sync`
   
   After the K-loop, store `acc_frag` into the output matrix `C` (FP32).

3. **`launchMatmul`** — choose a grid/block configuration such that exactly one warp handles each 16×16 output tile, then launch `matmulTensorCore`.

### Constraints

- Matrix dimensions `M`, `N`, `K` are guaranteed to be multiples of 16.
- Do not use `cudaMallocManaged`; use explicit `cudaMemcpy`.

---

## Testing

### Build

```bash
nvcc -arch=sm_80 -o student  student.cu
nvcc -arch=sm_80 -o solution solution.cu
```

### Run

```bash
./student   # prints timing and max-absolute-error vs CPU reference
./solution
```

### Expected output (example, 1024×1024×1024)

```
Matrix size: M=1024 N=1024 K=1024
Tensor Core kernel time : 0.83 ms
CPU reference time      : 3412.7 ms
Max absolute error      : 0.031250   (threshold 0.1) --> PASS
```
---