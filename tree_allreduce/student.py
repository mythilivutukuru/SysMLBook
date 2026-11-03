import multiprocessing as mp
import random
import sys
from turtle import up


def _send(queue: mp.Queue, value: float):
    """Blocking put onto `queue`."""
    queue.put(value)


def _recv(queue: mp.Queue) -> float:
    """Blocking get from `queue`."""
    return queue.get()


PARENT_OF = {0: -1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
CHILDREN_OF = {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [], 4: [], 5: [], 6: []}

ROOT = 0
LEAVES = [3, 4, 5, 6]
N_NODES = 7


def tree_worker(
    node_id: int,
    init_val: float,
    # Queues for upward communication (child -> parent).
    # up_send_q   : this node uses this to send its value UP to its parent.
    # up_recv_qs  : dict {child_id: Queue} this node reads from its children.
    up_send_q,  # mp.Queue | None  (None for root – has no parent)
    up_recv_qs: dict,  # {child_id -> mp.Queue}  (empty for leaves)
    # Queues for downward communication (parent -> child).
    # down_recv_q  : this node reads the broadcast value FROM its parent.
    # down_send_qs : dict {child_id: Queue} this node sends the broadcast DOWN.
    down_recv_q,  # mp.Queue | None  (None for root – has no parent)
    down_send_qs: dict,  # {child_id → mp.Queue}  (empty for leaves)
    result_dict,  # shared manager dict; write final value here
    barrier: mp.Barrier,
):
    val = init_val  # this node's accumulator

    # Phase 1 – Reduce  (bottom-up)
    # Goal: after Phase 1, the root (node 0) holds the sum of all 7 initial values.
    # A node that has children must first wait for each child to send its (partially-reduced) value, then accumulate those into `val`.
    # After accumulation (or immediately for leaves), a node sends `val` up to its parent – except the root, which has no parent.

    barrier.wait()
    if node_id == ROOT:
        print(f"\nPhase 1: Reduce  (bottom-up)")
        print(f"  Initial values: { {i: init_val for i in range(N_NODES)} }")

    # TODO
    # Accumulate from children (skip if leaf).
    # For each child_id in CHILDREN_OF[node_id]:
    #   1. Receive a float from up_recv_qs[child_id].
    #   2. Add the received value to `val`.

    # TODO
    # Send accumulated value to parent (skip if root).
    # If up_send_q is not None (i.e. this node is NOT the root):
    #   Send `val` via up_send_q.

    barrier.wait()
    if node_id == ROOT:
        print(f"\n  Root (Node 0) now holds global sum = {val:.1f}")
        print(f"\nPhase 2: Broadcast  (top-down)")

    # Phase 2 – Broadcast  (top-down)
    # Goal: every node ends with `val` == global sum.
    # The root already holds the global sum; it sends it down to each of its children.
    # Internal nodes receive the global sum from their parent, overwrite `val`, then forward it down to each of their children.
    # Leaves receive the global sum from their parent and overwrite `val`.  They have no children so they stop there.

    # TODO
    # Receive the global sum from parent (skip if root).

    # TODO
    # Forward the global sum to children (skip if leaf).

    result_dict[node_id] = val
    barrier.wait()


def run_tree_all_reduce(seed: int = 42):
    random.seed(seed)

    # Each node gets a random initial float value.
    init_vals = {i: float(random.randint(1, 10)) for i in range(N_NODES)}
    expected_sum = sum(init_vals.values())

    print(f"Tree All-Reduce  |  N_NODES={N_NODES}")
    for i in range(N_NODES):
        print(f"  Node {i} initial value : {int(init_vals[i])}")
    print(f"  Expected global sum  : {expected_sum:.1f}")

    edges = [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2)]  # (child, parent)

    up_q = {edge: mp.Queue() for edge in edges}  # keyed (child, parent)
    down_q = {(p, c): mp.Queue() for (c, p) in edges}  # keyed (parent, child)

    manager = mp.Manager()
    result_dict = manager.dict()
    barrier = mp.Barrier(N_NODES)

    processes = []
    for node_id in range(N_NODES):
        parent = PARENT_OF[node_id]
        children = CHILDREN_OF[node_id]
        up_send_q = up_q[(node_id, parent)] if parent != -1 else None
        up_recv_qs = {c: up_q[(c, node_id)] for c in children}
        down_recv_q = down_q[(parent, node_id)] if parent != -1 else None
        down_send_qs = {c: down_q[(node_id, c)] for c in children}

        p = mp.Process(
            target=tree_worker,
            args=(
                node_id,
                init_vals[node_id],
                up_send_q,
                up_recv_qs,
                down_recv_q,
                down_send_qs,
                result_dict,
                barrier,
            ),
            name=f"Node-{node_id}",
        )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("Correctness Check")

    all_ok = True
    for node_id in range(N_NODES):
        got = result_dict[node_id]
        ok = abs(got - expected_sum) < 1e-6
        status = "✓" if ok else "✗  FAIL"
        role = (
            "root" if node_id == ROOT else "leaf" if node_id in LEAVES else "internal"
        )
        print(
            f"  {status}  Node {node_id} ({role:8s}): got {got:.1f},  expected {expected_sum:.1f}"
        )
        if not ok:
            all_ok = False

    if all_ok:
        print(
            f"  ✓  ALL {N_NODES} nodes hold the correct global sum: {expected_sum:.1f}"
        )
    else:
        print("  ✗  Correctness check FAILED – see mismatches above.")
        sys.exit(1)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run_tree_all_reduce(seed=seed)
