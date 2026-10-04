# Decode-Priority Scheduling in Nano-vLLM

LLM inference engines run in a loop: at every step the **scheduler** picks a batch of sequences, the **model runner** executes it, and the engine updates state. Because the GPU runs either a *prefill* batch (processing whole prompts) or a *decode* batch (one new token per sequence) in a single step, the scheduler must decide which kind of work goes first whenever both are available.

## Background

[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) is a small, readable reimplementation of the core ideas behind vLLM: continuous batching, paged KV cache, prefix caching, tensor parallelism and CUDA graphs.

Its scheduler currently follows a **prefill-first** policy:

- If there are sequences in the `waiting` queue that fit in the token budget and the KV cache, they are prefilled right away, and no decode step happens in that iteration.
- Only when nothing can be prefilled does the scheduler run a decode step for the `running` sequences.
- If the KV cache runs out during decode, sequences are **preempted**: their blocks are freed and they go back to the `waiting` queue to be re-prefilled later.

Prefill-first tends to maximize the number of sequences in flight and gives new requests a low time-to-first-token (TTFT), but it **stalls requests that are already generating**, since every newly arrived prompt delays their next token.

In this assignment you will implement the opposite policy, **decode-first**.

## Task

Modify the scheduler so that **decode requests always take priority over prefill requests**.

Concretely, in each call to `schedule()`:

1. If there is at least one sequence in the `running` queue, schedule a **decode** step for the running sequences (subject to `max_num_seqs` and KV cache availability).
2. Only if there are **no running sequences** should the scheduler admit sequences from the `waiting` queue and schedule a **prefill** step.
3. The function must still return `(scheduled_seqs, is_prefill)` with the same meaning as before, so that `LLMEngine.step()` and `ModelRunner` work without changes.

You should only need to edit the `schedule()` method in `scheduler.py`. Do not change the `BlockManager`, `ModelRunner` or `Sequence` classes.

### Things to think about

These are the places where a naive rewrite will break. Make sure your implementation handles them.

- **Preemption can empty the running queue.** During decode, if there are not enough free blocks, the current code preempts sequences from the tail of `running` (and, as a last resort, the sequence itself). In decode-first mode it is possible that *every* running sequence gets preempted in a single call, leaving nothing to decode. Your scheduler must not return an empty batch; it needs to fall through to the prefill path in that case.
- **Preempted sequences go back to `waiting`.** Check that they are re-admitted correctly once resources become available, and that they are not lost or duplicated.
- **Per-step counters.** `num_seqs` and `num_batched_tokens` are computed per step. Think about whether the prefill admission logic still respects `max_num_seqs` and `max_num_batched_tokens` correctly after your change.
- **Termination.** The generation loop must still finish for every request, including when the KV cache is small enough to force preemption.
- **Queue ordering.** Decode scheduling should not reorder running sequences in a way that causes starvation of some of them.
