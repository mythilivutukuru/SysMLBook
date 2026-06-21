# KV Caching in nanoGPT

Modern transformer-based language models generate text autoregressively, producing one token at a time. In a naive implementation, at each generation step, the model recomputes attention over the entire sequence generated so far. This results in redundant computation and poor inference performance.

## Background 
**Key–Value (KV) caching** is a standard optimization technique that avoids this redundancy by storing previously computed key and value tensors from the attention mechanism and reusing them for subsequent decoding steps.

In this question, you will extend the NanoGPT codebase to support KV caching and evaluate its impact on inference speed.

[nanoGPT](https://github.com/karpathy/nanoGPT) is a minimal, lightweight implementation of GPT-style transformer models by Andrej Karpathy, designed for easy training and fine-tuning of medium-sized language models. It strips away unnecessary complexity while remaining faithful to the original GPT architecture.

You have been given the following files in the this sub-directory:

- `model.py` - a working implementation of NanoGPT (without KV caching)
- `evaluate.py` - an evaluation script to test correctness and performance
- `sample.py` - a sample file for you to debug and test your implementation

## Task

Your task is to modify `model.py` so that KV caching is correctly implemented.

To achieve this, you will need to modify the relevant parts of the model so that:

- The generation procedure is able to maintain and reuse a KV cache across decoding steps (this needs to be done only when the flag `enable_kv_caching` is set to `True`).
- The GPT model can optionally accept cached values from previous steps.
- Each transformer block can receive cached information and return updated cached values.
- The attention mechanism is adapted to correctly operate when past keys and values are reused.

## Testing

Make sure that `evaluate.py`, `model.py` and `configurator.py` are present in the same directory. You can start the evaluation script through `python3 evaluate.py`. It runs four test prompts with increasing generation lengths:

| Test Case | Prompt | Max Tokens |
|---|---|---|
| 1 | "Once upon a time" | 10 |
| 2 | "The quick brown fox" | 50 |
| 3 | "In a galaxy far, far away" | 100 |
| 4 | "To be or not to be" | 500 |

For each test case, the script measures:

- Runtime without KV caching
- Runtime with KV caching
- Whether the outputs match
- Speedup factor

Here is how the output might roughly look like:

```
...
================================================================================
Test Case 1
Prompt: 'Once upon a time'
Max new tokens: 10
================================================================================

 Outputs Match: True

Timing:
  Without KV Cache: 0.3229s
  With KV Cache:    0.2390s
  Speedup:          1.35x
...

```

After implementing KV caching correctly, you should observe:

- No difference in generated text between cached and non-cached modes.
- Noticeable inference speed improvements, especially for long sequences.

You may not see significant speedup for very short outputs, but for longer generations (e.g., 100-500 tokens), the improvement should be substantial (atleast 1.5-2X).