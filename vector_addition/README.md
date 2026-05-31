# Assignment: CUDA Vector Addition Kernel

In this assignment, you will implement a parallelized vector addition kernel in CUDA. You will write the GPU kernel that adds two float arrays element-wise and verify correctness against a CPU reference implementation.

---

## Background

### CUDA Thread Hierarchy

CUDA executes code on thousands of threads simultaneously. Threads are grouped into **blocks**, and blocks are grouped into a **grid**.

```
Grid
 └── Block [0]   Block [1]   Block [2]   ...
      └── Thread 0..N-1
```

Each thread has a unique identity through built-in variables:

| Variable | Type | Description |
|---|---|---|
| `threadIdx.x` | `uint3` | Thread index *within* its block (0 … blockDim.x − 1) |
| `blockIdx.x`  | `uint3` | Block index *within* the grid (0 … gridDim.x − 1) |
| `blockDim.x`  | `dim3`  | Number of threads per block |
| `gridDim.x`   | `dim3`  | Number of blocks in the grid |

### Computing a Global Thread Index

For 1-D problems the **global index** uniquely identifies which array element a thread should process:

```
globalIndex = threadIdx.x + blockDim.x * blockIdx.x
```

Because the total number of threads launched may exceed the array length `N`, every kernel must guard against out-of-bounds access:

```c
if (globalIndex < N) { /* safe to read/write arrays */ }
```

### Kernel Launch Syntax

```c
kernel<<<numBlocks, threadsPerBlock>>>(args...);
```

The number of blocks needed to cover `N` elements is:

```c
int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
```

This ceiling-division ensures every element is covered even when `N` is not a multiple of `threadsPerBlock`.

---

## Task

Open `student.cu`. You will find three `TODO` comments inside the `vecAdd` kernel and the `main` function. Fill in each one:

### TODO 1 — Compute the global thread index

Inside `vecAdd`, calculate the unique index for this thread across the entire grid.

**Hint:** use `threadIdx.x`, `blockDim.x`, and `blockIdx.x`.

### TODO 2 — Guard against out-of-bounds access

Before writing to the output array, ensure `globalIndex` is within the valid range `[0, N)`.

### TODO 3 — Perform the addition

Write the sum of the corresponding elements of `A` and `B` into `C`.

---

After completing the kernel, also read through `main()` to understand how memory is allocated on the GPU, how data is copied to the device, how the kernel is launched, and how results are copied back.

You are **not** required to modify `main()` or `cpuVecAdd()`.

---

## Building

```bash
nvcc -o vecadd student.cu
```

Requires CUDA Toolkit ≥ 10.0 and a CUDA-capable GPU.

---

## Testing

The program runs automatically after the kernel launch. It:

1. Initializes two arrays `A` and `B` with known values on the CPU.
2. Copies them to the GPU, runs your kernel, and copies the result back.
3. Computes the expected result using `cpuVecAdd()`.
4. Compares every element; if all match within a small tolerance (`1e-5`), it prints:

```
PASSED: GPU result matches CPU result.
```

Otherwise it prints the index of the first mismatch and exits with a non-zero code:

```
FAILED at index 42: GPU=... CPU=...
```

### Expected Output (correct solution)

```
Launching kernel: 4 blocks x 256 threads = 1024 threads total
PASSED: GPU result matches CPU result.
```

---

## Tips

- A common bug is forgetting the bounds check — without it, threads beyond index `N−1` will read/write garbage memory.
- `blockDim.x * blockIdx.x` gives the starting index of the current block; adding `threadIdx.x` gives the offset within the block.
- CUDA errors are silent unless you check return codes. The helper macro `CUDA_CHECK` in the student file will abort with a message if any API call fails.