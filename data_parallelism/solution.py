import argparse
import os
import threading
import time
from multiprocessing import Barrier, Process, Queue

import numpy as np

from mlp import MLP

INPUT_DIM = 8
HIDDEN_DIM = 16
OUTPUT_DIM = 4
BATCH_SIZE = 32
EPOCHS = 3
LR = 0.01
NUM_GPUS = 4
BYTES_PER_FLOAT = 4
BANDWIDTH_GBPS = 10.0
BYTES_PER_GB = 1e9
LATENCY_BASE_SEC = 0.0001


def simulated_transfer_time(num_floats: int) -> float:
    num_bytes = num_floats * BYTES_PER_FLOAT
    return LATENCY_BASE_SEC + num_bytes / (BANDWIDTH_GBPS * BYTES_PER_GB)


class NetworkThread:

    def __init__(self, n: int, grad_size: int):
        self.n = n
        self.grad_size = grad_size
        self.send_queue: Queue = Queue()
        self.recv_queues: list = [Queue() for _ in range(n)]
        self._sum_buffer = np.zeros(grad_size, dtype=np.float64)
        self._contributions = 0
        self._sum_ready = threading.Event()
        self._lock = threading.Lock()
        self.total_bytes_transferred = 0.0
        self.total_sim_time = 0.0
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True
        )
        self._recv_thread.start()
        self._dispatch_thread.start()

    def get_queues(self):
        return self.send_queue, self.recv_queues

    def _recv_loop(self):
        while True:
            # TODO: Block-wait on self.send_queue to get next item.
            #   Break out of the loop if None is received.
            item = self.send_queue.get()
            if item is None:
                break

            rank, grad = item

            sim_t = simulated_transfer_time(len(grad))
            time.sleep(sim_t)

            with self._lock:
                # TODO: Accumulate grad into _sum_buffer,
                #   increment _contributions, and update stats counters.
                self._sum_buffer += grad
                self._contributions += 1
                self.total_bytes_transferred += len(grad) * BYTES_PER_FLOAT
                self.total_sim_time += sim_t

                # TODO: If all N gradients have arrived, signal _sum_ready.
                if self._contributions == self.n:
                    self._sum_ready.set()

    def _dispatch_loop(self):
        while True:
            # TODO: Block until _sum_ready is set.
            self._sum_ready.wait()

            with self._lock:
                # TODO: Compute the averaged gradient.
                avg_grad = self._sum_buffer / self.n

                # TODO: Unblock every GPU process by pushing
                #   avg_grad to each rank's recv_queue.
                for q in self.recv_queues:
                    q.put(avg_grad.copy())

                # TODO: Reset sum_buffer, contributions counter,
                #   and clear the _sum_ready event for the next round.
                self._sum_buffer[:] = 0.0
                self._contributions = 0
                self._sum_ready.clear()

    def shutdown(self):
        self.send_queue.put(None)

    def report(self):
        print("\n" + "=" * 50)
        print("  Communication Statistics")
        print("=" * 50)
        print(f"  Total data transferred : {self.total_bytes_transferred / 1e6:.4f} MB")
        print(f"  Total simulated time   : {self.total_sim_time:.6f} s")
        print("=" * 50 + "\n")


def gpu_worker(
    rank: int,
    n_gpus: int,
    send_queue: Queue,
    recv_queues: list,
    barrier: Barrier,
    epochs: int,
):
    print(f"[GPU {rank}] Starting  (pid={os.getpid()})")

    model = MLP(seed=42)
    rng = np.random.default_rng(rank + 100)

    for epoch in range(epochs):
        X = rng.standard_normal((BATCH_SIZE, INPUT_DIM)).astype(np.float64)
        Y = rng.standard_normal((BATCH_SIZE, OUTPUT_DIM)).astype(np.float64)

        logits, cache = model.forward(X)
        loss_val = model.loss(logits, Y)
        local_grad = model.backward(cache, Y)

        print(f"[GPU {rank}] Epoch {epoch+1}/{epochs}  loss={loss_val:.5f}")

        # TODO: Send local_grad to the NetworkThread.
        #   Push the tuple (rank, local_grad) onto send_queue.
        send_queue.put((rank, local_grad))

        # TODO: Receive the averaged gradient.
        #   Block on recv_queues[rank] until NetworkThread pushes the result.
        avg_grad = recv_queues[rank].get()

        # TODO: Apply the averaged gradient to the model.
        model.apply_gradients(avg_grad)

        barrier.wait()

    print(f"[GPU {rank}] Done.")


def coordinator(n_gpus: int = NUM_GPUS, epochs: int = EPOCHS):
    print(f"\n{'='*50}")
    print(f"  Data Parallel MLP  |  N = {n_gpus} GPUs")
    print(f"  Epochs             : {epochs}")
    print(f"  Model param count  : {MLP().grad_size()}")
    print(f"{'='*50}\n")

    net = NetworkThread(n=n_gpus, grad_size=MLP().grad_size())

    send_queue, recv_queues = net.get_queues()

    barrier = Barrier(n_gpus)
    workers = []

    for rank in range(n_gpus):
        p = Process(
            target=gpu_worker,
            args=(rank, n_gpus, send_queue, recv_queues, barrier, epochs),
            daemon=False,
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    net.shutdown()
    net.report()


# ======================================================================
# TESTING: PLEASE DO NOT MODIFY BELOW THIS LINE
# ======================================================================


def run_baseline(n_gpus: int = NUM_GPUS, epochs: int = EPOCHS) -> dict:
    print("\n" + "=" * 50)
    print("  Running Serial Baseline")
    print("=" * 50)

    models = [MLP(seed=42) for _ in range(n_gpus)]
    rngs = [np.random.default_rng(rank + 100) for rank in range(n_gpus)]

    results = {"losses": [], "final_param_norm": None}

    for epoch in range(epochs):
        epoch_losses = []
        grads = []

        for rank in range(n_gpus):
            X = rngs[rank].standard_normal((BATCH_SIZE, INPUT_DIM)).astype(np.float64)
            Y = rngs[rank].standard_normal((BATCH_SIZE, OUTPUT_DIM)).astype(np.float64)

            logits, cache = models[rank].forward(X)
            loss_val = models[rank].loss(logits, Y)
            local_grad = models[rank].backward(cache, Y)

            epoch_losses.append(loss_val)
            grads.append(local_grad)
            print(
                f"[Baseline GPU {rank}] Epoch {epoch+1}/{epochs}  loss={loss_val:.5f}"
            )

        avg_grad = np.mean(grads, axis=0)

        for rank in range(n_gpus):
            models[rank].apply_gradients(avg_grad)

        results["losses"].append(epoch_losses)

    final_norm = float(np.linalg.norm(models[0].params_flat()))
    results["final_param_norm"] = final_norm
    print(f"\n[Baseline] Final param norm: {final_norm:.6f}")
    return results


def gpu_worker_instrumented(
    rank, n_gpus, send_queue, recv_queues, barrier, epochs, result_q
):
    model = MLP(seed=42)
    rng = np.random.default_rng(rank + 100)
    losses = []

    for epoch in range(epochs):
        X = rng.standard_normal((BATCH_SIZE, INPUT_DIM)).astype(np.float64)
        Y = rng.standard_normal((BATCH_SIZE, OUTPUT_DIM)).astype(np.float64)

        logits, cache = model.forward(X)
        loss_val = model.loss(logits, Y)
        local_grad = model.backward(cache, Y)

        losses.append(loss_val)
        send_queue.put((rank, local_grad))
        avg_grad = recv_queues[rank].get()
        model.apply_gradients(avg_grad)
        barrier.wait()

    final_norm = float(np.linalg.norm(model.params_flat()))
    result_q.put({"rank": rank, "losses": losses, "final_param_norm": final_norm})


def run_distributed_and_capture(n_gpus: int = NUM_GPUS, epochs: int = EPOCHS) -> dict:
    result_queue = Queue()

    net = NetworkThread(n=n_gpus, grad_size=MLP().grad_size())
    send_queue, recv_queues = net.get_queues()
    barrier = Barrier(n_gpus)

    workers = []
    for rank in range(n_gpus):
        p = Process(
            target=gpu_worker_instrumented,
            args=(rank, n_gpus, send_queue, recv_queues, barrier, epochs, result_queue),
            daemon=False,
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    net.shutdown()

    all_results = [result_queue.get() for _ in range(n_gpus)]
    all_results.sort(key=lambda r: r["rank"])
    return all_results


def correctness_check(n_gpus: int = NUM_GPUS, epochs: int = EPOCHS):
    print("\n" + "#" * 50)
    print("  CORRECTNESS CHECK")
    print("#" * 50)

    baseline = run_baseline(n_gpus=n_gpus, epochs=epochs)
    dist = run_distributed_and_capture(n_gpus=n_gpus, epochs=epochs)

    print("\n" + "=" * 50)
    print("  Comparison: Baseline vs Distributed")
    print("=" * 50)

    all_passed = True
    TOL = 1e-8

    for epoch in range(epochs):
        for rank in range(n_gpus):
            b_loss = baseline["losses"][epoch][rank]
            d_loss = dist[rank]["losses"][epoch]
            diff = abs(b_loss - d_loss)
            status = "PASS" if diff < TOL else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(
                f"  Epoch {epoch+1} GPU {rank}: "
                f"baseline={b_loss:.6f}  dist={d_loss:.6f}  "
                f"diff={diff:.2e}  [{status}]"
            )

    print()
    b_norm = baseline["final_param_norm"]
    for r in dist:
        d_norm = r["final_param_norm"]
        diff = abs(b_norm - d_norm)
        status = "PASS" if diff < TOL else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(
            f"  Final param norm GPU {r['rank']}: "
            f"baseline={b_norm:.6f}  dist={d_norm:.6f}  "
            f"diff={diff:.2e}  [{status}]"
        )

    print()
    if all_passed:
        print("  *** ALL CHECKS PASSED — distributed run matches baseline ***")
    else:
        print("  *** SOME CHECKS FAILED — review your AllReduce implementation ***")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP MLP Assignment")
    parser.add_argument("--n_gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run correctness check against serial baseline",
    )
    args = parser.parse_args()

    if args.check:
        correctness_check(n_gpus=args.n_gpus, epochs=args.epochs)
    else:
        coordinator(n_gpus=args.n_gpus, epochs=args.epochs)
