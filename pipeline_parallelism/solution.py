import multiprocessing as mp

import numpy as np

IN_F = 8
HIDDEN = 16
OUT_F = 4


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)


def mse_loss_grad(pred, target):
    return 2.0 * (pred - target) / pred.size


def init_weights(in_f, out_f, seed):
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0 / in_f)
    W = rng.standard_normal((out_f, in_f)) * scale
    b = np.zeros(out_f)
    return W, b


def linear_fwd(x, W, b):
    return x @ W.T + b


def linear_bwd(dz, x, W):
    dW = dz.T @ x
    db = dz.sum(axis=0)
    dx = dz @ W
    return dx, dW, db


TAG_ACT = "activation"
TAG_OUTPUT = "output"
TAG_LOSS_GRAD = "loss_grad"
TAG_GRAD_ACT = "grad_act"
TAG_GRADS = "grads"


def run_serial(x, target, seed=42):
    print("\n" + "=" * 64)
    print("  SERIAL BASELINE  (single process, no IPC)")
    print("=" * 64)

    W1, b1 = init_weights(IN_F, HIDDEN, seed)
    W2, b2 = init_weights(HIDDEN, OUT_F, seed + 1)

    # Forward
    z1 = linear_fwd(x, W1, b1)
    act1 = relu(z1)
    z2 = linear_fwd(act1, W2, b2)
    loss = mse_loss(z2, target)
    print(f"  [Serial] Forward  done | loss = {loss:.10f}")

    # Backward
    dz2 = mse_loss_grad(z2, target)
    dact1, dW2, db2 = linear_bwd(dz2, act1, W2)
    dz1 = dact1 * relu_grad(z1)
    _, dW1, db1 = linear_bwd(dz1, x, W1)
    print(f"  [Serial] Backward done")
    print(
        f"  [Serial] norm(dW1)={np.linalg.norm(dW1):.10f}  norm(dW2)={np.linalg.norm(dW2):.10f}"
    )

    return dict(loss=loss, dW1=dW1, db1=db1, dW2=dW2, db2=db2)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-0 worker  (owns Layer 1)
# ─────────────────────────────────────────────────────────────────────────────
def gpu0_worker(x, q_to_gpu1, q_from_gpu1, q_to_main, seed):
    W1, b1 = init_weights(IN_F, HIDDEN, seed)

    # --- FORWARD ---
    print("  [GPU-0] Layer-1 forward ...")

    # TODO — Forward pass of Layer 1
    # Compute the pre-activation z1 by passing x through the linear layer,
    # then apply ReLU to get the post-activation act1.
    z1 = linear_fwd(x, W1, b1)
    act1 = relu(z1)

    print(f"  [GPU-0] act1 ready. " f"Sending to GPU-1 via Queue ...")

    # TODO - Send the activation AND the pre-activation cache to GPU-1 using q_to_gpu1.put().
    q_to_gpu1.put((TAG_ACT, act1, z1))

    # --- BACKWARD ---
    # Block here until GPU-1 finishes its own backward and sends back
    # dL/d(act1) — the gradient of the loss w.r.t. our output activation.
    tag, dact1 = q_from_gpu1.get()
    assert tag == TAG_GRAD_ACT
    print(f"  [GPU-0] Received dact1 from GPU-1  (norm={np.linalg.norm(dact1):.6f})")

    # TODO — Backward pass of Layer 1
    # We have dL/d(act1) in hand.  To continue back-propagation through Layer 1
    # we must first pass the gradient through the ReLU gate, then through the
    # linear layer. Perform ReLU backward first, then linear backward to get the parameter gradients.
    dz1 = dact1 * relu_grad(z1)
    _, dW1, db1 = linear_bwd(dz1, x, W1)

    q_to_main.put((TAG_GRADS, "gpu0", dict(dW1=dW1, db1=db1)))
    print(f"  [GPU-0] Backward done. norm(dW1)={np.linalg.norm(dW1):.10f}")


# ─────────────────────────────────────────────────────────────────────────────
# GPU-1 worker  (owns Layer 2)
# ─────────────────────────────────────────────────────────────────────────────
def gpu1_worker(target, q_from_gpu0, q_to_gpu0, q_to_main, q_from_main, seed):
    W2, b2 = init_weights(HIDDEN, OUT_F, seed + 1)

    # --- FORWARD ---
    # Block here until GPU-0 finishes Layer-1 and puts the activation on the
    # queue.
    tag, act1, _z1 = q_from_gpu0.get()
    assert tag == TAG_ACT
    print(f"  [GPU-1] Received act1 from GPU-0  (norm={np.linalg.norm(act1):.6f})")

    # TODO — Forward pass of Layer 2 + hand-off to Main
    # Compute the network output z2 by passing act1 through the second linear
    # layer. Then send z2 to the Main process so it can compute the loss.
    # The message must be a tuple tagged TAG_OUTPUT so Main can identify it.
    z2 = linear_fwd(act1, W2, b2)
    print(f"  [GPU-1] Layer-2 forward done. " f"Sending output to Main ...")
    q_to_main.put((TAG_OUTPUT, z2))

    # --- BACKWARD ---
    # Block until Main has computed dL/d(z2) and sent it back.
    # This is the entry point of the backward pass on GPU-1: the loss gradient
    # w.r.t. the final layer output, computed by the coordinator.
    tag, dz2 = q_from_main.get()
    assert tag == TAG_LOSS_GRAD
    print(f"  [GPU-1] Received loss-grad from Main  (norm={np.linalg.norm(dz2):.6f})")

    # TODO — Backward pass of Layer 2 + cross-GPU gradient send
    # Use linear_bwd to propagate dz2 through Layer 2.  This yields dact1, dW2, and db2.
    # Two sends are required:
    #   1. Send (TAG_GRADS, "gpu1", {dW2, db2}) to q_to_main.
    #   2. Send (TAG_GRAD_ACT, dact1)           to q_to_gpu0
    dact1, dW2, db2 = linear_bwd(dz2, act1, W2)
    q_to_main.put((TAG_GRADS, "gpu1", dict(dW2=dW2, db2=db2)))
    q_to_gpu0.put((TAG_GRAD_ACT, dact1))
    print(
        f"  [GPU-1] Backward done. norm(dW2)={np.linalg.norm(dW2):.10f}. "
        f"Sent dact1 to GPU-0."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline coordinator
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(x, target, seed=42):
    print("\n" + "=" * 64)
    print("  PIPELINE PARALLEL  (2 x multiprocessing.Process + Queues)")
    print("=" * 64)

    ctx = mp.get_context("spawn")
    q_0_to_1 = ctx.Queue()
    q_1_to_0 = ctx.Queue()
    q_0_to_main = ctx.Queue()
    q_1_to_main = ctx.Queue()
    q_main_to_1 = ctx.Queue()

    p0 = ctx.Process(
        target=gpu0_worker,
        args=(x, q_0_to_1, q_1_to_0, q_0_to_main, seed),
        name="GPU-0",
    )
    p1 = ctx.Process(
        target=gpu1_worker,
        args=(target, q_0_to_1, q_1_to_0, q_1_to_main, q_main_to_1, seed),
        name="GPU-1",
    )

    p0.start()
    p1.start()
    print("  [Main ] Spawned GPU-0 and GPU-1 processes")

    # Receive model output from GPU-1
    tag, out = q_1_to_main.get()
    assert tag == TAG_OUTPUT
    print(
        f"  [Main ] Received model output from GPU-1  (norm={np.linalg.norm(out):.6f})"
    )

    # Compute loss and gradient
    loss = mse_loss(out, target)
    dout = mse_loss_grad(out, target)
    print(f"  [Main ] Loss = {loss:.10f}  |  sending loss-grad to GPU-1 ...")
    q_main_to_1.put((TAG_LOSS_GRAD, dout))

    # Collect gradients from both workers
    grads = {}
    _, src1, g1 = q_1_to_main.get()  # GPU-1 sends its grads first (layer 2 bwd is done)
    grads[src1] = g1
    _, src0, g0 = q_0_to_main.get()  # GPU-0 grads arrive after it receives dact1
    grads[src0] = g0

    p0.join()
    p1.join()

    dW1 = grads["gpu0"]["dW1"]
    db1 = grads["gpu0"]["db1"]
    dW2 = grads["gpu1"]["dW2"]
    db2 = grads["gpu1"]["db2"]

    print(f"\n  [Main ] Pipeline complete")
    print(
        f"  [Main ] norm(dW1)={np.linalg.norm(dW1):.10f}  norm(dW2)={np.linalg.norm(dW2):.10f}"
    )

    return dict(loss=loss, dW1=dW1, db1=db1, dW2=dW2, db2=db2)


# ─────────────────────────────────────────────────────────────────────────────
# Correctness check
# ─────────────────────────────────────────────────────────────────────────────
def check_correctness(serial, pipeline, tol=1e-10):
    print("\n" + "=" * 64)
    print("  CORRECTNESS CHECK  (serial vs pipeline)")
    print("=" * 64)

    all_ok = True
    for key in ("loss", "dW1", "db1", "dW2", "db2"):
        s, p = serial[key], pipeline[key]
        if np.ndim(s) == 0:
            diff = abs(float(s) - float(p))
            note = f"serial={s:.8f}  pipeline={p:.8f}  diff={diff:.2e}"
        else:
            diff = float(np.abs(s - p).max())
            note = f"max_abs_diff = {diff:.2e}"
        ok = diff < tol
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {key:6s}  {note}")

    banner = (
        "ALL PASS  -- pipeline matches serial exactly!"
        if all_ok
        else "MISMATCH DETECTED"
    )
    print(f"\n  Overall --> {banner}")
    return all_ok


if __name__ == "__main__":
    SEED = 42
    BATCH = 4

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((BATCH, IN_F))
    target = rng.standard_normal((BATCH, OUT_F))

    print("PIPELINE PARALLELISM SIMULATION")
    print(f"\n  Input  shape : {x.shape}   Target shape : {target.shape}")
    print(
        f"  Model        : Layer1(Linear {IN_F}->{HIDDEN} + ReLU) -> Layer2(Linear {HIDDEN}->{OUT_F})"
    )

    serial_results = run_serial(x, target, seed=SEED)
    pipeline_results = run_pipeline(x, target, seed=SEED)
    check_correctness(serial_results, pipeline_results)
