import multiprocessing as mp
import random
import sys


def _send(queue: mp.Queue, data):
    """Put data onto a queue (blocking)."""
    queue.put(data)


def _recv(queue: mp.Queue):
    """Get data from a queue (blocking)."""
    return queue.get()


def gpu_worker(
    rank: int,  # this GPU's id  (0 … N_GPUS-1)
    n_gpus: int,  # total number of GPUs
    n_ranks: int,  # number of rank-slots per GPU (== n_gpus)
    init_data: list,  # initial data[i] is the value for rank-slot i on this GPU
    send_queue: mp.Queue,  # queue to the RIGHT neighbour
    recv_queue: mp.Queue,  # queue from the LEFT  neighbour
    result_dict,  # shared dict to write final result
    barrier: mp.Barrier,  # synchronisation barrier (all GPUs wait here)
):

    # 'buf' holds n_ranks slots.  Initially it is a copy of init_data.
    # During Scatter-Reduce these slots are partially reduced;
    # during All-Gather they are filled with the final global sums.
    buf = list(init_data)  # length == n_ranks

    # Phase 1: Scatter-Reduce  (N-1 rounds)
    # Goal: after N-1 rounds, buf[rank] on this GPU holds the fully-reduced (summed) value for rank-slot `rank`.
    # In each round every GPU:
    #   1. Sends one chunk (identified by the current send-slot) to its right
    #      neighbour.
    #   2. Receives one chunk from its left neighbour.
    #   3. Adds the received value INTO its local copy of that chunk.

    barrier.wait()  # all GPUs start Phase 1 together
    if rank == 0:
        print("Phase 1: Scatter-Reduce")

    for step in range(n_gpus - 1):

        # TODO
        # Determine which slot this GPU should SEND in this step.
        # Intuition: in step 0 each GPU sends its own slot (rank).
        #            In step 1 it forwards the chunk it just received, which
        #            originated one GPU to its left, i.e. slot (rank-1).
        #            In general, in step s the send slot is (rank - s) % n_gpus.
        # Compute `send_slot` using the formula above.
        send_slot = (rank - step) % n_gpus

        # TODO
        # Determine which slot this GPU should RECEIVE (accumulate into)
        # in this step.
        # The chunk arriving from the LEFT neighbour originated one GPU
        # further left, so its slot index is (rank - step - 1) % n_gpus.
        # Compute `recv_slot` using the formula above.
        recv_slot = (rank - step - 1) % n_gpus

        # TODO
        # Send buf[send_slot] to the right neighbour via `send_queue`.
        # Then receive a value from the left neighbour via `recv_queue`.
        _send(send_queue, buf[send_slot])
        incoming = _recv(recv_queue)

        # TODO
        # Accumulate: add `incoming` into buf[recv_slot].
        # This is the "reduce" part of scatter-reduce: each GPU accumulates
        # partial sums until one GPU holds the complete sum for each slot.
        buf[recv_slot] += incoming

        print(
            f"  [GPU {rank}] scatter step {step}: "
            f"sent slot {send_slot} (val={buf[send_slot]:.1f}), "
            f"recv slot {recv_slot}, buf={[round(v,1) for v in buf]}"
        )

    # After Phase 1: GPU `g` holds the fully-reduced sum at buf[g].
    barrier.wait()
    if rank == 0:
        print("Phase 2: All-Gather")

    # Phase 2: All-Gather  (N-1 rounds)
    # Goal: propagate each GPU's fully-reduced slot around the entire ring
    #       so every GPU ends up with all n_ranks reduced values.
    #
    # In each round every GPU:
    #   1. Sends one chunk (the "already finalised" one) to its right neighbour.
    #   2. Receives a finalised chunk from its left neighbour.
    #   3. Overwrites its local copy of that chunk with the received value — it is already the true global sum.

    for step in range(n_gpus - 1):

        # TODO
        # Determine which slot this GPU should SEND in this all-gather step.
        # In step 0 we send the slot we own after scatter-reduce, which is
        # `rank` itself.  In step 1 we forward the chunk we just received,
        # which is (rank + 1) % n_gpus.  In general: (rank + 1 - step) % n_gpus.
        # Compute `send_slot` for the all-gather phase.
        send_slot = (rank + 1 - step) % n_gpus

        # TODO
        # Determine the receive slot: one step further around the ring from
        # the send slot, i.e. (rank - step) % n_gpus.
        # Compute `recv_slot` for the all-gather phase.
        recv_slot = (rank - step) % n_gpus

        # TODO
        # Send buf[send_slot] to the right neighbour, receive from the left
        # neighbour, and overwrite buf[recv_slot] with the received value.
        _send(send_queue, buf[send_slot])
        incoming = _recv(recv_queue)
        buf[recv_slot] = incoming

        print(
            f"  [GPU {rank}] gather  step {step}: "
            f"sent slot {send_slot}, recv slot {recv_slot}, "
            f"buf={[round(v,1) for v in buf]}"
        )

    result_dict[rank] = buf
    barrier.wait()  # wait for every GPU to finish


# ======================================================================
# TESTING: PLEASE DO NOT MODIFY BELOW this LINE
# ======================================================================


def run_ring_all_reduce(n_gpus: int, seed: int = 42):
    random.seed(seed)
    n_ranks = n_gpus

    init = [
        [float(random.randint(1, 10)) for _ in range(n_ranks)] for _ in range(n_gpus)
    ]

    expected = [sum(init[g][r] for g in range(n_gpus)) for r in range(n_ranks)]

    print(f"Ring All-Reduce  |  n_gpus={n_gpus}  n_ranks={n_ranks}")
    print("-" * 60)
    for g in range(n_gpus):
        print(f"  GPU {g} initial data : {[int(v) for v in init[g]]}")
    print(f"  Expected sum      : {[int(v) for v in expected]}")

    queues = [mp.Queue() for _ in range(n_gpus)]

    manager = mp.Manager()
    result_dict = manager.dict()

    barrier = mp.Barrier(n_gpus)

    processes = []
    for g in range(n_gpus):
        send_q = queues[g]
        recv_q = queues[(g - 1) % n_gpus]
        p = mp.Process(
            target=gpu_worker,
            args=(
                g,
                n_gpus,
                n_ranks,
                init[g],
                send_q,
                recv_q,
                result_dict,
                barrier,
            ),
            name=f"GPU-{g}",
        )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"\n{'='*60}")
    print("Correctness Check")
    print(f"{'='*60}")

    all_ok = True
    for g in range(n_gpus):
        got = result_dict[g]
        for r in range(n_ranks):
            if abs(got[r] - expected[r]) > 1e-6:
                print(f"  FAIL  GPU {g} slot {r}: got {got[r]}, expected {expected[r]}")
                all_ok = False
        print(f"  GPU {g} result : {[int(v) for v in got]}")

    if all_ok:
        print(
            f"\n  ✓  ALL {n_gpus} GPUs hold the correct global sum: {[int(v) for v in expected]}"
        )
    else:
        print("\n  ✗  Correctness check FAILED – see mismatches above.")
        sys.exit(1)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_ring_all_reduce(n_gpus=N)
