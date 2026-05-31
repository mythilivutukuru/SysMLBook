# MLP Forward Pass with Streams

In this assignment, you will implement the forward pass of a two-layer Multi-Layer Perceptron (MLP) using CUDA. You will write two custom kernels — one for a fused matrix-multiply + ReLU activation, and one for a plain matrix-multiply output layer — and orchestrate them using **CUDA streams** for potential overlap with memory transfers.

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

A naïve CUDA kernel assigns one thread per output element. A tile-based shared memory kernel (extra credit) is significantly faster.

### ReLU Activation

Applied element-wise after the first MatMul:

```
ReLU(x) = max(0, x)
```

In your kernel you will fuse this into the MatMul itself — compute the dot product, add the bias, then clamp to zero before writing to global memory.

### Pinned (Page-Locked) Host Memory

Ordinary `malloc` memory can be swapped to disk by the OS at any time. The GPU's DMA engine requires memory whose physical pages are guaranteed to stay resident.

`cudaMallocHost` allocates page-locked memory directly:

```c
float *h_X;
cudaMallocHost(&h_X, bytes);   // pinned — DMA can read/write directly
// ...
cudaFreeHost(h_X);             // must use cudaFreeHost, NOT free()
```

### CUDA Streams

A **CUDA stream** is a sequence of GPU operations that execute in issue order relative to each other, but may overlap with operations in *other* streams.

Key API calls you will use:

```c
cudaStream_t stream;
cudaStreamCreate(&stream);                        // create
kernel<<<grid, block, 0, stream>>>(...);          // launch into stream
cudaMemcpyAsync(dst, src, bytes, kind, stream);   // async copy in stream
cudaStreamSynchronize(stream);                    // wait for stream
cudaStreamDestroy(stream);                        // cleanup
```

Why streams here? In a real inference server you would overlap the **H2D copy** of a new batch with the **kernel execution** of the current batch, or overlap the **D2H copy** of results with the next batch's computation. Pinned memory is the prerequisite that makes this overlap physically possible.

---

## Task

You are given `student.cu`. It contains:

- All data structure definitions and host-side setup code (do **not** modify these).
- A CPU reference implementation (`mlp_cpu`) for correctness checking.
- Two kernel stubs with `TODO` comments:
  - `matmul_relu_kernel` — fused MatMul + ReLU for the hidden layer.
  - `matmul_kernel` — plain MatMul + bias for the output layer.
- A launch wrapper stub `mlp_forward_gpu` with `TODO` comments for stream creation, async memory copies, kernel launches, and synchronization.
- A `main` function that runs both the CPU and GPU paths and compares outputs.

### Your Tasks

1. **`matmul_relu_kernel`**
   - Each thread computes one element of the output matrix.
   - Compute the dot product, add the corresponding bias, apply ReLU, and write to output.

2. **`matmul_kernel`**
   - Same as above but **without** the ReLU clamp.

3. **`mlp_forward_gpu`**
   - Create a CUDA stream.
   - Asynchronously copy all inputs (X, W1, b1, W2, b2) to the device using the stream.
   - Launch `matmul_relu_kernel` in the stream to produce H.
   - Launch `matmul_kernel` in the stream to produce Y.
   - Asynchronously copy Y back to host in the stream.
   - Synchronize and destroy the stream.

### Constraints

- Do **not** use `cudaMemcpy` (blocking); use `cudaMemcpyAsync` with your stream.
- Do **not** use cuBLAS or any external math library.
- Grid and block dimensions are provided as hints in the stub — you may adjust them.
- Keep all other code (CPU reference, main, error checking) unchanged.

---

## Testing

The `main` function automatically:

1. Runs the CPU reference forward pass.
2. Runs your GPU forward pass.
3. Compares every element of the output matrix Y element-wise with a tolerance of **1e-3**.
4. Prints `PASS` if all elements match, or `FAIL` with the first mismatched index and values.

Sample expected output on success:

```
[CPU] MLP forward pass complete.
[GPU] MLP forward pass complete.
Checking correctness...
Max absolute error: 0.000023
PASS: GPU output matches CPU reference.
```
---