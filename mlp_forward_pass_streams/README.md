# MLP Forward Pass with 4 Streams

In this assignment, you will implement the forward pass of a two-layer Multi-Layer Perceptron (MLP) using CUDA. You will write two custom kernels — one for a fused matrix-multiply + ReLU activation, and one for a plain matrix-multiply output layer — and orchestrate them using **four concurrent CUDA streams** to overlap host-to-device transfers with kernel execution.

---

## Background

### What is an MLP?

A Multi-Layer Perceptron is the simplest form of a feedforward neural network. Given an input matrix **X**, the two-layer forward pass computes:

```
H = ReLU(X  @ W1 + b1)   ← hidden layer
Y = H @ W2 + b2           ← output layer (no activation)
```

where:

| Symbol | Shape          | Description                        |
|--------|----------------|------------------------------------|
| X      | (N, D_in)      | Batch of N input samples           |
| W1     | (D_in, D_h)    | Weight matrix, layer 1             |
| b1     | (D_h,)         | Bias vector, layer 1               |
| H      | (N, D_h)       | Hidden activations after ReLU      |
| W2     | (D_h, D_out)   | Weight matrix, layer 2             |
| b2     | (D_out,)       | Bias vector, layer 2               |
| Y      | (N, D_out)     | Final output logits                |

### Matrix Multiply (MatMul)

Each output element `C[i][j]` is the dot product of row `i` of **A** and column `j` of **B**:

```
C[i][j] = sum_k  A[i][k] * B[k][j]  +  bias[j]
```

A naïve CUDA kernel assigns one thread per output element.

### ReLU Activation

Applied element-wise after the first MatMul:

```
ReLU(x) = max(0, x)
```

In your kernel you will fuse this into the MatMul itself — compute the dot product, add the bias, then clamp to zero before writing to global memory.

### Pinned (Page-Locked) Host Memory

`cudaMallocHost` allocates page-locked memory that the GPU's DMA engine can read/write directly, enabling asynchronous (overlapped) transfers:

```c
float *h_X;
cudaMallocHost(&h_X, bytes);   // pinned — DMA can read/write directly
cudaFreeHost(h_X);             // must use cudaFreeHost, NOT free()
```

### CUDA Streams and Overlap

A **CUDA stream** is a sequence of GPU operations that execute in order relative to each other but may overlap with operations in *other* streams.

```c
cudaStream_t stream;
cudaStreamCreate(&stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(dst, src, bytes, kind, stream);
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

**Why 4 streams?** The GPU has an independent DMA copy engine. When the batch is split across 4 streams, the copy engine can transfer X for stream `s+1` while the SMs execute kernels for stream `s`, cutting end-to-end latency compared to a single serialised stream.

---

## Task

### Design

The batch (`N = 64` rows) is divided into `NUM_STREAMS = 4` equal chunks of `CHUNK = 16` rows. Stream `s` owns rows `[s*CHUNK, (s+1)*CHUNK)`.

#### Phase 0 — Weight Upload (one-time, synchronous)

W1, b1, W2, b2 are **shared and read-only** across all chunks. They are copied to the device **once** with blocking `cudaMemcpy` before any stream starts.

#### Phase 1 — Per-Stream Pipeline (all 4 streams in flight)

For each stream `s`:

| Step | Operation | Detail |
|------|-----------|--------|
| (a) | H2D async copy | `CHUNK` rows of `X` → `d_X + s*CHUNK*D_IN` |
| (b) | `matmul_relu_kernel` | Slice of `X` × `W1` + `b1`, ReLU → slice of `H` |
| (c) | `matmul_kernel` | Slice of `H` × `W2` + `b2` → slice of `Y` |
| (d) | D2H async copy | `CHUNK` rows of `Y` ← `d_Y + s*CHUNK*D_OUT` |

Within a single stream, (a)→(b)→(c)→(d) are strictly ordered. Across streams, the GPU overlaps the H2D copy of stream `s+1` with the kernel execution of stream `s`.

```
Stream 0:  [H2D X₀]──[ReLU]──[MatMul]──[D2H Y₀]
Stream 1:       [H2D X₁]──[ReLU]──[MatMul]──[D2H Y₁]
Stream 2:            [H2D X₂]──[ReLU]──[MatMul]──[D2H Y₂]
Stream 3:                 [H2D X₃]──[ReLU]──[MatMul]──[D2H Y₃]
           ──────────────────────────────────────────────────▶ time
```

#### Phase 2 — Synchronise and Destroy

`cudaStreamSynchronize` + `cudaStreamDestroy` called for all 4 streams.

### Your Tasks

Both kernels operate on **slices** of the full batch — `rows_A = CHUNK`, not `N`.

**`matmul_relu_kernel`** (TODOs 1–5):
- TODO 1: Compute `col = threadIdx.x + blockIdx.x * blockDim.x`, `row = threadIdx.y + blockIdx.y * blockDim.y`
- TODO 2: Guard — return if `row >= rows_A || col >= cols_B`
- TODO 3: Accumulate dot product over `k = 0..inner-1`: `acc += A[row*inner+k] * B[k*cols_B+col]`
- TODO 4: Add `bias[col]`
- TODO 5: Write `fmaxf(acc, 0.0f)` to `C[row*cols_B+col]`

**`matmul_kernel`** (TODOs 6–9): identical except no `fmaxf` in TODO 9 — just write `acc + bias[col]`.

#### Launch Wrapper (TODOs 10–21)

**Phase 0 — weight upload:**
- TODOs 10–13: `cudaMemcpy` (blocking) for W1, b1, W2, b2

**Phase 1 — stream setup:**
- TODO 14: `cudaStream_t streams[NUM_STREAMS]`; `cudaStreamCreate` for each
- TODO 15: `dim3 block(16,16)`, `dim3 grid_h` (sized to `D_H` × `CHUNK`), `dim3 grid_out` (sized to `D_OUT` × `CHUNK`)

**Per-stream loop body:**
- TODO 16: Compute `row_off`, `x_off`, `h_off`, `y_off`; derive slice pointers
- TODO 17: `cudaMemcpyAsync` — X slice H2D into `streams[s]`
- TODO 18: Launch `matmul_relu_kernel` with `rows_A = CHUNK` into `streams[s]`
- TODO 19: Launch `matmul_kernel` with `rows_A = CHUNK` into `streams[s]`
- TODO 20: `cudaMemcpyAsync` — Y slice D2H into `streams[s]`

**Phase 2:**
- TODO 21: `cudaStreamSynchronize` + `cudaStreamDestroy` for each stream

### Constraints

- In the per-stream loop, use **`cudaMemcpyAsync`** (not blocking `cudaMemcpy`) so transfers can overlap across streams.
- For weight uploads in Phase 0, use blocking **`cudaMemcpy`** — they happen once before any stream starts.
- Do **not** use cuBLAS or any external math library.
- Keep the kernels, CPU reference, and `main` unchanged in structure.

---

## Testing

The `main` function automatically:

1. Runs the CPU reference forward pass.
2. Runs your GPU forward pass (4-stream).
3. Compares every element of the output matrix Y element-wise with a tolerance of **1e-3**.
4. Prints `PASS` if all elements match, or `FAIL` with the first mismatched index.

Expected output on success:

```
[CPU] MLP forward pass complete.
[GPU] MLP forward pass complete.
Checking correctness...
Max absolute error: 0.000023
PASS: GPU output matches CPU reference.
```

---