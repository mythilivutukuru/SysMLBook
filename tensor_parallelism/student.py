import multiprocessing as mp

import numpy as np

# ---------------------------------------------------------------------------
# Serial (CPU) reference implementation
# ---------------------------------------------------------------------------


def serial_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute C = A @ B on a single process.

    Args:
        A: Matrix of shape [M, K]
        B: Matrix of shape [K, N]

    Returns:
        C: Matrix of shape [M, N]
    """
    return A @ B


# ---------------------------------------------------------------------------
# Helper: shard B across `num_gpus` along the column (N) dimension
# ---------------------------------------------------------------------------


def shard_columns(B: np.ndarray, num_gpus: int) -> list[np.ndarray]:
    """
    Split matrix B along its column dimension into `num_gpus` equal shards.

    Example:
        B has shape [K, N].  With num_gpus=2 you get two shards each of
        shape [K, N//2].

    Args:
        B:        Weight matrix of shape [K, N].  Assume N % num_gpus == 0.
        num_gpus: Number of shards to produce.

    Returns:
        List of `num_gpus` arrays, each of shape [K, N // num_gpus].

    # TODO:  Implement the column sharding.
    #          Hint: np.array_split or np.split along axis=1.
    """
    
    pass
    


# ---------------------------------------------------------------------------
# Worker function — runs inside each child process ("GPU")
# ---------------------------------------------------------------------------


def gpu_worker(
    rank: int,
    A: np.ndarray,
    B_shard: np.ndarray,
    result_queue: mp.Queue,
) -> None:
    """
    Simulate one GPU doing its share of the matrix multiply.

    Each worker:
      1. Computes its local partial result   C_local = A @ B_shard
      2. Sends (rank, C_local) back to the parent via `result_queue`

    Args:
        rank:         GPU index (0 or 1).
        A:            The full activation matrix [M, K] (replicated on every GPU).
        B_shard:      This GPU's column shard of B [K, N // num_gpus].
        result_queue: Multiprocessing Queue used to return the result.

    # TODO:  Compute C_local = A @ B_shard and put (rank, C_local) into
    #          result_queue so the parent can collect and assemble results.
    """
    print(f"[GPU {rank}] Starting computation on shard of shape {B_shard.shape}")
    
    C_local = 

    

    print(f"[GPU {rank}] Done — local result shape {C_local.shape}")


# ---------------------------------------------------------------------------
# Coordinator — shards data, spawns workers, gathers results
# ---------------------------------------------------------------------------


def tensor_parallel_matmul(
    A: np.ndarray, B: np.ndarray, num_gpus: int = 2
) -> np.ndarray:
    """
    Compute C = A @ B using `num_gpus` parallel processes.

    Steps:
      1. Shard B into `num_gpus` column shards.
      2. Spawn one process per GPU and have each compute its local partial result.
      3. Collect partial results from all GPUs (via a Queue).
      4. Concatenate the partial results in the correct order to produce C.

    Args:
        A:        Activation matrix [M, K].
        B:        Weight matrix  [K, N].  Assume N % num_gpus == 0.
        num_gpus: Number of simulated GPUs (processes).

    Returns:
        C: Matrix of shape [M, N].
    """
    # ------------------------------------------------------------------
    # Step 1 — shard B
    # ------------------------------------------------------------------
    # TODO:  Call your shard_columns() function to get a list of B shards.
    
    B_shards = 
    

    print(
        f"\n[Coordinator] Sharded B {B.shape} into {num_gpus} shards, "
        f"each {B_shards[0].shape}"
    )

    # ------------------------------------------------------------------
    # Step 2 — spawn worker processes
    # ------------------------------------------------------------------
    result_queue: mp.Queue = mp.Queue()
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(rank, A, B_shards[rank], result_queue),
        )
        p.start()
        processes.append(p)

    # ------------------------------------------------------------------
    # Step 3 — collect results (inter-process communication / AllGather)
    # ------------------------------------------------------------------
    # TODO:  Receive `num_gpus` (rank, C_local) pairs from result_queue.
    #          Store them in a dict keyed by rank so you can sort later.
    
    partial_results: dict[int, np.ndarray] = {}


    

    # Wait for all processes to finish cleanly
    for p in processes:
        p.join()

    # ------------------------------------------------------------------
    # Step 4 — concatenate in rank order to reconstruct C
    # ------------------------------------------------------------------
    # TODO:  Concatenate the partial results along axis=1 (columns) in
    #          ascending rank order to obtain the full C matrix.
    #          Hint: np.concatenate([partial_results[r] for r in sorted(...)], axis=1)
    
    C = 
    

    return C


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Matrix dimensions
    M, K, N = 512, 256, 1024  # A: [M,K]  B: [K,N]  C: [M,N]
    NUM_GPUS = 2

    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)

    print("=" * 60)
    print("  Tensor Parallelism — Column Parallel MatMul")
    print("=" * 60)
    print(f"  A shape : {A.shape}")
    print(f"  B shape : {B.shape}")
    print(f"  Simulated GPUs : {NUM_GPUS}")

    print("\n--- Serial (CPU) ---")
    C_serial = serial_matmul(A, B)
    print(f"Serial result shape : {C_serial.shape}")

    print("\n--- Tensor Parallel ---")
    C_parallel = tensor_parallel_matmul(A, B, num_gpus=NUM_GPUS)
    print(f"\n[Coordinator] Parallel result shape : {C_parallel.shape}")

    print("\n--- Correctness Check ---")
    if np.allclose(C_parallel, C_serial, atol=1e-4):
        print("Correctness check PASSED ✓")
    else:
        max_diff = np.max(np.abs(C_parallel - C_serial))
        print(f"Correctness check FAILED ✗  (max |diff| = {max_diff:.6f})")

    print("\n--- Summary ---")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
