#Tensor Parallelism

Large language models distribute weights across multiple devices using **Tensor Parallelism**. In linear layers, the weight matrix $B$ is sharded along its column dimension so that each device owns an independent slice and computes its portion of output in parallel; the partial results are then concatenated to reconstruct the full output.

In this assignment, you will implement the the missing pieces inside the functions of `student.py`.

---

## Background

### Setup

Given matrices

$$
A \in \mathbb{R}^{M \times K}, \qquad B \in \mathbb{R}^{K \times N},
$$

the goal is to compute $C = AB \in \mathbb{R}^{M \times N}$ using `num_gpus` parallel worker processes.

$B$ is partitioned into `num_gpus` equal column shards (assume $N \bmod \texttt{num\_gpus} = 0$):

$$
B = \bigl[\,B^{(0)} \;\big|\; B^{(1)} \;\big|\; \cdots \;\big|\;
           B^{(\texttt{num\_gpus}-1)}\,\bigr],
\qquad B^{(r)} \in \mathbb{R}^{K \times (N/\texttt{num\_gpus})}.
$$

Each worker $r$ receives the full matrix $A$ and its own shard $B^{(r)}$, computes the local product $C^{(r)} = A B^{(r)}$, and returns the result. The coordinator concatenates the partial results in rank order:

$$
C = \bigl[\,C^{(0)} \;\big|\; C^{(1)} \;\big|\; \cdots \;\big|\;
           C^{(\texttt{num\_gpus}-1)}\,\bigr].
$$

| Symbol | Meaning |
|---|---|
| $M, K, N$ | Matrix dimensions |
| `num_gpus` | Number of simulated GPU worker processes |
| $B^{(r)}$ | Column shard assigned to worker $r$ |
| $C^{(r)} = A B^{(r)}$ | Local partial result from worker $r$ |
| `result_queue` | `mp.Queue` used by workers to return results |

### Processes and Queues

The simulation spawns `num_gpus` Python child processes. Communication between the coordinator and workers uses a single shared `multiprocessing.Queue`:

| Direction | Queue | Payload |
|---|---|---|
| worker $\to$ coordinator | `result_queue` | `(rank, C_local)` |

Each worker puts a `(rank, C_local)` tuple onto `result_queue` so the coordinator can collect and order all partial results.

### Algorithm

#### Step 1 — Shard $B$ (`shard_columns`)

Split $B$ along `axis=1` into `num_gpus` equal sub-matrices.

- Input: $B \in \mathbb{R}^{K \times N}$, integer `num_gpus`.
- Output: list of `num_gpus` arrays, each of shape $[K,\; N/\texttt{num\_gpus}]$.
- Hint: `np.array_split` or `np.split` along `axis=1`.

#### Step 2 — Worker Computation (`gpu_worker`)

Each worker process executes:

1. Compute $C_{\text{local}} = A \,@\, B^{(r)}$ (shape $[M,\; N/\texttt{num\_gpus}]$).
2. Put `(rank, C_local)` onto `result_queue`.

#### Step 3 — Coordinate and Gather (`tensor_parallel_matmul`)

The coordinator:

1. Calls `shard_columns` to obtain the list of $B$ shards.
2. Spawns one `mp.Process` per rank, passing the corresponding shard.
3. Receives `num_gpus` `(rank, C_local)` pairs from `result_queue` and stores them in a `dict` keyed by rank.
4. Concatenates the partial results in ascending rank order along `axis=1` to produce the full matrix $C$.

## Task

Open `student.py` and complete every `# TODO` block.

**Constraints:**

- Do not modify any code outside the `# TODO` blocks.
- Do not add new queues or processes beyond those already provided.
- Use only `mp.Queue` for inter-process communication; do not use shared memory or pipes.
- Assume $N \bmod \texttt{num\_gpus} = 0$ (no uneven sharding needed).

## Testing

The `main` function acts as the correctness checker. It generates random matrices $A$ and $B$, runs both the serial and parallel implementations, and verifies:

$$
\max_{i,j} \bigl|C_{\text{parallel}} - C_{\text{serial}}\bigr|_{ij}
\;\leq\; 10^{-4}.
$$

Run the tests with:

```
python3 student.py
```
