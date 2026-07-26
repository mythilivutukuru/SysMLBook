# One Element per Thread Tiling

Naive GPU matrix multiplication re-reads rows and columns of the input
matrices directly from global memory for every multiply-accumulate
operation, which wastes a huge amount of memory bandwidth. This assignment
introduces **shared-memory tiling**, the first step toward high-performance
GPU matmul kernels: each thread block cooperatively stages small tiles of
$A$ and $B$ into fast on-chip shared memory, and every thread in the block
reuses those tiles many times before they are evicted.

In this version, each thread is responsible for computing exactly **one**
element of the output matrix $C$. Later assignments build on this design by
having each thread compute several output elements (register blocking), but
the goal here is to understand the core tiling loop:

1. Cooperatively load a tile of $A$ and a tile of $B$ into shared memory.
2. Synchronize so every thread sees the fully loaded tiles.
3. Accumulate partial dot-product contributions from the tiles into a
   per-thread register.
4. Synchronize again before loading the next tile.
5. After all tiles have been processed, write the accumulated result to
   global memory.

---

## Background

### Problem Setup

Given two square matrices $A, B \in \mathbb{R}^{N \times N}$, compute:

$$
C = A \times B,
$$

where

$$
C[i,j] = \sum_{k=0}^{N-1} A[i,k] \cdot B[k,j],
\qquad 0 \le i,j < N.
$$

All matrices are stored in **row-major flattened format**.

### Kernel Design Overview

Let `TILE` be the shared-memory tile dimension. Each thread block is
launched with a 2D block of dimensions:

$$
\mathtt{TILE} \times \mathtt{TILE}
$$

and is responsible for computing a $\mathtt{TILE} \times \mathtt{TILE}$
tile of the output matrix $C$. Each individual thread inside the block
computes exactly **one** output element.

### Shared Memory Layout

A single dynamically allocated shared-memory buffer is used, split into two
halves:

```cpp
extern __shared__ float s[];
float* sA = s;                // TILE * TILE floats
float* sB = s + TILE*TILE;    // TILE * TILE floats
```

Both `sA` and `sB` are treated as row-major $\mathtt{TILE} \times \mathtt{TILE}$
arrays.

### Thread Responsibilities

Let `(localRow, localCol)` denote a thread's coordinates inside its block
(`threadIdx.y`, `threadIdx.x`), and let `(globalRow, globalCol)` denote the
coordinates of the output element it owns in $C$:

$$
\begin{aligned}
\mathtt{globalRow} &= \mathtt{blockIdx.y} \times \mathtt{blockDim.y} + \mathtt{threadIdx.y}, \\
\mathtt{globalCol} &= \mathtt{blockIdx.x} \times \mathtt{blockDim.x} + \mathtt{threadIdx.x}.
\end{aligned}
$$

Each thread must:

- Maintain a single accumulator register (e.g. `float acc = 0.0f;`).
- Load exactly one element of $A$ into `sA` and one element of $B$ into
  `sB` per tile iteration.
- Accumulate into `acc`.
- Write the final result back to global memory.

### Tiling Along the K Dimension

The multiplication proceeds over tiles along the shared $K$ dimension. The
number of tiles is:

$$
\texttt{numTiles} = \left\lceil \frac{N}{\mathtt{TILE}} \right\rceil.
$$

For each tile index $t$, the block cooperatively loads one tile of $A$ and
one tile of $B$ into shared memory.

#### Loading Tile of A

Each thread loads the element of $A$ at global row `globalRow` and global
column `t * TILE + localCol` into `sA[localRow][localCol]`. If either index
falls outside the bounds of the matrix (which can happen when $N$ is not an
exact multiple of `TILE`), the thread should store `0.0f` instead so that
out-of-bounds contributions do not corrupt the sum.

#### Loading Tile of B

Similarly, each thread loads the element of $B$ at global row
`t * TILE + localRow` and global column `globalCol` into
`sB[localRow][localCol]`, again zero-padding out-of-bounds accesses.

After loading both tiles, call:

```cpp
__syncthreads();
```

so that every thread in the block can safely read the tiles that other
threads loaded.

#### Compute Phase

Each thread walks across the shared dimension of the tile (`k = 0 ... TILE-1`)
and accumulates:

$$
\mathtt{acc} \;+\!= \mathtt{sA}[\mathtt{localRow}][k] \times \mathtt{sB}[k][\mathtt{localCol}].
$$

Then call `__syncthreads()` before proceeding to the next tile, so that no
thread starts overwriting `sA`/`sB` for the next iteration while other
threads are still reading the current tile.

### Final Write-Back

After all tiles have been processed, each thread writes its accumulated
`acc` value into `C[globalRow * N + globalCol]`, guarded by a bounds check
in case `globalRow` or `globalCol` is outside `[0, N)`.

## Task

You are given the file `student.cu` in this directory. It contains:

- `main()` (testing infrastructure),
- `matmul_gpu()` (kernel launch wrapper), and
- The signature of the CUDA kernel that you must implement.

Implement the kernel:

```cpp
__global__ void matmul_tiled_element_kernel(
    const float* A,
    const float* B,
          float* C,
    int N,
    int TILE
)
```

- `A, B` = input matrices in row-major flattened format ($N \times N$).
- `C` = output matrix in row-major flattened format ($N \times N$).
- `N` = side length of all three matrices.
- `TILE` = shared-memory tile size; the block dimension is
  $\mathtt{TILE} \times \mathtt{TILE}$.

**Requirements:**

- Use `extern __shared__` memory and partition it into `sA` and `sB`.
- Compute `globalRow`, `globalCol`, `localRow`, and `localCol`.
- Declare and zero-initialize a per-thread accumulator.
- Loop over all $K$-dimension tiles.
- Perform cooperative loading into shared memory, zero-padding
  out-of-bounds elements.
- Synchronize correctly before and after the compute phase.
- Accumulate the dot-product contribution from each tile.
- Write the final result to global memory with a bounds check.

## Running

```bash
nvcc -O3 -o matmul_tiled_element student.cu
./matmul_tiled_element <N> <TILE>
```

`N` is the matrix dimension and `TILE` is the shared-memory tile size (and
therefore the block dimension along each axis). `TILE` should evenly divide
typical block-size limits, e.g. try `TILE = 16` or `TILE = 32`.

## Testing

The provided test:

- Computes a CPU reference result.
- Launches your kernel via `matmul_gpu`.
- Compares results element-wise.

A test case receives PASS only if **all** elements satisfy the following
criteria:

$$
|C_{\text{CPU}} - C_{\text{GPU}}| < 10^{-3}.
$$
