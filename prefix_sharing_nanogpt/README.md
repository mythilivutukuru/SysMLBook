# Prefix Sharing in nanoGPT

In many real-world scenarios (such as chatbots), multiple requests often share a large, identical system prompt or context.

## Background

Naive KV caching recomputes the key-value pairs for this shared prefix for every single request in a batch. **Prefix sharing** is an optimization where the KV cache for the common prefix is computed only once and shared across all sequences in the batch, significantly reducing redundant computation.

You have been given the following files in this sub-directory:

- `model.py` - a working implementation of NanoGPT (without KV caching)
- `evaluate.py` - an evaluation script to test correctness and performance

When generating for a batch of sequences $[S_1, S_2, \dots, S_n]$ that all start with a common prefix $P$:

1. **Compute once:** The model processes $P$ and stores its key and value tensors $(K_p, V_p)$.
2. **Share:** For each sequence $S_i$, the generation starts from the end of $P$, initializing its local KV cache with the shared $(K_p, V_p)$.
3. **Diverge:** As each sequence generates unique tokens, their individual KV caches grow independently.

## Task

Your task is to modify the `generate` method and the underlying `GPT` model in `model.py` to detect and utilize shared prefixes.

To achieve this, you need to:

- **Find the Common Prefix:** Implement logic to identify the longest common sequence of tokens across all input prompts in a batch.
- **Compute Shared Cache:** Forward the common prefix through the model to obtain the initial `prefix_kvs`.
- **Autoregressive Generation:** This is similar to what exists already. Starting from the shared cache, generate tokens one by one.
- **Do not implement KV Caching:** You do not need to add the standard KV caching optimization for this question. You just need to store (and return) the KV cache for the prefix shared across all the sequences. During generation, only the prefix KV cache should be reused - do **NOT** cache the suffix or newly generated tokens. Each forward pass should recompute everything after the prefix. This gives an idea of how much prefix sharing alone improves over the original baseline.

### Generate Function Interface

Make sure your implementation aligns with the following interface:

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, enable_prefix_sharing=True):
    """
    Parameters
    ----------
    idx : List[torch.Tensor]
        Each tensor has shape [1, seq_len], where seq_len might be different
        across all the sequences

    max_new_tokens : int
        Number of new tokens to generate for each input sequence.
        The total output length will be (original_length + max_new_tokens).

    enable_prefix_sharing : bool
        Whether to enable the prefix sharing optimization.

    Returns
    -------
    outputs : List[torch.Tensor]
        Generated sequences including both original and new tokens.

    shared_prefix_kvs : List[Tuple[torch.Tensor, torch.Tensor]] or None
        Cached attention keys and values for the shared prefix.

        - If prefix sharing is active and a common prefix exists:
          Returns a list with one tuple per transformer layer.
          Each tuple contains (key_cache, value_cache) where:
            - key_cache:   Shape [1, n_head, prefix_len, head_size]
            - value_cache: Shape [1, n_head, prefix_len, head_size]

        - Returns None if:
          - enable_prefix_sharing is False
          - No common prefix exists
    """
```

### Hints

We have provided a few hints in `model.py` to get you started. Along with those, keep the following in mind:

- **Query/key length mismatch:** When using prefix KV cache sharing, the query and key sequences no longer have the same length. Let the shared prefix length be $P$ and the length of the current input be $T$. Queries correspond only to the $T$ newly provided tokens, while keys and values correspond to the entire sequence of length $P + T$. The causal mask must account for this offset.
- **Positional embeddings:** When prefix KV caches are present, positional embeddings must reflect the *true positions* of tokens in the full sequence, not just their local indices within the current input. If a cached prefix of length $P$ already exists and the model is given $T$ new tokens, those tokens occupy positions $[P, P+1, \dots, P+T-1]$, so the positional embedding must reflect this.

## Testing

Make sure that `evaluate.py`, `model.py` and `configurator.py` are present in the same directory. You can start the evaluation script through `python3 evaluate.py`.

The provided evaluation script verifies your implementation using a large system prompt (the shared prefix, roughly 540 tokens long), followed by different user queries. It measures:

- **Correctness:** The generated text must be identical to the non-shared implementation (the simple decoding already provided in the code).
- **Efficiency:** Time taken to generate tokens for the entire batch.
- **KV Cache Integrity:** The shape of the shared prefix cache must be `(1, n_head, T_prefix, head_dim)` for each of the transformer layers.

After implementing prefix sharing correctly, you should observe:

- No difference in generated text between prefix-sharing and non-shared modes.
- A valid shared prefix KV cache.
- For larger shared prefix lengths and a smaller number of new decode tokens, a significant speedup (at least **1.5x**).

Here is how the output might roughly look like:

```
...
--------------------------------------------------------------------------------
WITHOUT PREFIX SHARING
--------------------------------------------------------------------------------
Run 1: 13.657s
...
================================================================================
Overall KV cache validity:  VALID
================================================================================
...
================================================================================
All outputs match:  YES
================================================================================
...
```

## Notes

- Make sure you identify the longest shared prefix correctly (there should not exist a longer prefix that is common across all input sequence tensors). 
- You can safely assume that the total sequence length won't exceed 1024, so there is no need to modify `self.config.block_size`.
- All input sequences are tokenized using the same tokenizer. Each input tensor has shape `[1, seq_len]`. All tensors are on the same device. The batch size will be $\geq 1$.
- Use `.clone()` when distributing the shared prefix KV cache to individual sequences, to avoid unintended in-place modifications to the shared base.
- **Common pitfalls:** If your outputs don't match between prefix sharing enabled/disabled, the issue is likely in how you handle positional embeddings or causal attention masks, so make sure you use offsets correctly.
- Please ignore any bias-related warnings that come up while loading GPT-2 model weights.
