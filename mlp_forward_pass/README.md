# MLP Forward Pass

In this assignment, you will implement the forward pass of a two-layer Multi-Layer Perceptron (MLP) entirely on the GPU using CUDA. You will write two custom CUDA kernels — one for matrix multiplication and one for the ReLU activation — and wire them together in a launch wrapper. A CPU reference implementation is provided so you can verify correctness.

---

## Background

A **Multi-Layer Perceptron** is the simplest form of a feedforward neural network. Given an input matrix **X**, a two-layer MLP computes:

```
Z1 = X  @ W1 + b1       # Linear transform, layer 1
A1 = ReLU(Z1)           # Activation
Z2 = A1 @ W2 + b2       # Linear transform, layer 2
out = Z2                 # Final output (no activation on last layer)
```

Where `@` denotes matrix multiplication and `ReLU(x) = max(0, x)`.

### Matrix multiplication recap

For `C = A × B` where A is `(M × K)` and B is `(K × N)`, the output C is `(M × N)` and:

```
C[row][col] = sum over k of A[row][k] * B[k][col]
```

In a naive CUDA kernel, one thread computes exactly one `C[row][col]`.

---

## Task

Open **`student.cu`**. You will find:

- All host-side setup code already written (memory allocation, data generation, result comparison).
- A CPU reference implementation for correctness checking.
- **Three TODOs** for you to complete:

### TODO 1 — `matmul_kernel`

Write a CUDA kernel that computes `C = A × B`.

- `A` is `(M × K)`, `B` is `(K × N)`, `C` is `(M × N)` — all in **row-major** order.
- Launch one thread per output element.
- Each thread should compute the dot product of one row of A with one column of B.

### TODO 2 — `relu_kernel`

Write a CUDA kernel that applies ReLU element-wise to a matrix.

- The matrix has `rows × cols` elements stored contiguously in row-major order.
- Each thread handles one element: `x = max(0, x)`.

### TODO 3 — `mlp_forward` (launch wrapper)

Wire the two kernels together to perform the full two-layer forward pass:

1. Call `matmul_kernel` to compute `Z1 = X @ W1` then add bias `b1`.
2. Call `relu_kernel` to compute `A1 = ReLU(Z1)`.
3. Call `matmul_kernel` to compute `Z2 = A1 @ W2` then add bias `b2`.
4. Choose sensible grid/block dimensions for each launch.
5. Synchronise with `cudaDeviceSynchronize()` before returning.

> **Bias addition tip:** You may add the bias inside `matmul_kernel` by passing the bias vector and adding `bias[col]` to each output element, or you can write a separate small bias-add step — your choice.

---

## Testing

You need the CUDA toolkit (`nvcc`) installed.

```bash
# Compile student version
nvcc -o mlp_student student.cu

# Compile solution (after you're done, to compare)
nvcc -o mlp_solution solution.cu
```

Both binaries run the same self-contained test:

```
./mlp_student
./mlp_solution
```

The program will:

1. Generate random input `X` `(BATCH × IN_DIM)` and random weights/biases.
2. Run the CPU reference implementation.
3. Run your CUDA forward pass.
4. Compare every element of the output matrices with a tolerance of **1e-4**.
5. Print `PASSED` or `FAILED` (with the first mismatching value and its index).

### Expected output on success

```
Running 2-layer MLP forward pass...
  Batch=64  in=128  hidden=256  out=10
[CPU]  reference done.
[CUDA] kernel done.
Max absolute error: 0.000003
Result: PASSED
```
---

## Notes

- Use `threadIdx`, `blockIdx`, and `blockDim` to compute a 2-D `(row, col)` index.
- Guard every kernel with a bounds check (`if (row < M && col < N)`).
- A block size of `16×16` or `32×32` is a natural starting point.
- `ceil(M / BLOCK) = (M + BLOCK - 1) / BLOCK` gives you the number of blocks in one dimension.

---