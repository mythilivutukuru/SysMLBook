# Ring AllReduce

Distributed training of large neural networks requires aggregating gradients computed independently on each GPU so that every device ends up with the same global sum. One standard algorithm for this is **Ring AllReduce**: the GPUs are arranged in a logical ring, and values circulate around the ring in two phases until every GPU holds the fully-reduced result.

In this assignment you will implement the missing computation inside `gpu_worker` (in `student.py`), a function that each simulated GPU executes as its own process.

---

## Background

### Setup

| Symbol | Meaning |
|---|---|
| $N$ | Number of GPUs (= number of rank-slots) |
| $g$ | This GPU's rank, $g \in \{0,\dots,N-1\}$ |
| `buf[r]` | GPU $g$'s local copy of rank-slot $r$ |
| `init[g][r]` | Initial value for slot $r$ on GPU $g$ (for example gradients) |

Each GPU owns a buffer `buf` of length $N$. Initially `buf[r] = init[g][r]`. After a successful All-Reduce every GPU should hold, at every slot $r$:

$$
\texttt{buf}[r] = \sum_{g=0}^{N-1} \texttt{init}[g][r].
$$

### Processes and Queues

The simulation spawns $N$ Python processes communicating through `multiprocessing.Queue` objects, which act as the inter-GPU communication channel (analogous to NCCL send/recv on real hardware).

| Process | Variable | Responsibility |
|---|---|---|
| `GPU-g` | `send_queue` | Queue to the **right** neighbour (GPU $(g+1) \bmod N$) |
| `GPU-g` | `recv_queue` | Queue from the **left** neighbour (GPU $(g-1) \bmod N$) |

Two helper functions are already provided:

- `_send(queue, value)` — puts `value` onto `queue`.
- `_recv(queue)` — blocks until a value is available, then returns it.

---

## Algorithm: Two-Phase Ring AllReduce

### Phase 1 — ScatterReduce ($N-1$ steps)

The goal is to let one GPU accumulate the complete global sum for each slot. In each step $s \in \{0,\dots,N-2\}$, every GPU simultaneously:

1. **Sends** `buf[send_slot]` to its **right** neighbour via `send_queue`, where

   $$
   \texttt{send\_slot} = (g - s) \bmod N.
   $$

2. **Receives** a value from its **left** neighbour via `recv_queue` into slot

   $$
   \texttt{recv\_slot} = (g - s - 1) \bmod N.
   $$

3. **Accumulates:** `buf[recv_slot] += incoming`.

After $N-1$ steps, GPU $g$ holds the fully-reduced (global) sum at `buf[g]`; all other slots are partial sums only.

### Phase 2 — AllGather ($N-1$ steps)

The goal is to broadcast each GPU's fully-reduced value around the ring. In each step $s \in \{0,\dots,N-2\}$, every GPU simultaneously:

1. **Sends** `buf[send_slot]` to its right neighbour, where

   $$
   \texttt{send\_slot} = (g + 1 - s) \bmod N.
   $$

2. **Receives** a finalised value from its left neighbour into

   $$
   \texttt{recv\_slot} = (g - s) \bmod N.
   $$

3. **Overwrites:** `buf[recv_slot] ← incoming`.

After $N-1$ steps every slot on every GPU holds the correct global sum.

### Execution Timeline

1. All GPUs synchronise at a `barrier` and start Phase 1.
2. Each GPU executes $N-1$ ScatterReduce steps concurrently.
3. All GPUs synchronise at a second `barrier` and start Phase 2.
4. Each GPU executes $N-1$ AllGather steps concurrently.
5. Each GPU writes its final `buf` into `result_dict[rank]`.
6. All GPUs synchronise at a final `barrier` before exiting.

---

## Task

### Task Description

Open `student.py` and complete the `# TODO` blocks inside `gpu_worker`.

### Constraints

- Do not modify any code outside the eight `# TODO` blocks.
- Do not add new queues or processes.
- Use only the provided `_send` and `_recv` helpers for communication.

---

## Testing

The `run_ring_all_reduce` function at the bottom of the file acts as the correctness checker. It initialises random integer buffers, runs the multi-process simulation, and verifies that every GPU's final buffer matches the expected element-wise sum. Run the tests with:

```bash
python3 student.py          # default: 4 GPUs
python3 student.py 6        # custom: 6 GPUs
```
