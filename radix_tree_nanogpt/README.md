# Radix Tree in nanoGPT

In this assignment, you will implement radix tree in nanoGPT.

## Background

Some inputs may share text in their prompts, for which we need not repeat the KV cache computation. For example, if two input prompts share a common prefix, we can perform the KV cache computation for the shared prefix first, and reuse this KV cache when processing the unique suffixes of the input prompts, and when generating the remaining tokens in the output.

This type of pattern is common in real-world applications, e.g., when there is a large common system prompt, followed by individual user prompts.

## Task

Your task is to modify your KV caching implementation (`model.py`, and `sample.py` as needed) so that KV caches of shared prefixes are reused across the prompts in a batch.

You can use any data structure of your choice, e.g., a [radix tree](https://en.wikipedia.org/wiki/Radix_tree), to track shared prefixes across prompts. You can assume that a batch of input prompts is passed to the `generate` function (you may modify the `sample.py` script for this purpose), so that you can identify shared prefixes and compute KV caches in a suitable order that enables good reuse.

### Example: Radix Tree

Given a batch of tokens:

```
[1, 2, 3, 4]
[1, 2, 3, 5]
[1, 2, 6]
```

You can compute a radix tree as follows:

```
root
 └── [1, 2]
        ├── [3]
        │    ├── [4]
        │    └── [5]
        └── [6]
```

Each node in the radix tree stores a contiguous subset of tokens, and the corresponding KV cache (when computed), so that you can retrieve the KV cache corresponding to shared prefixes easily, by traversing the tree and concatenating the KV caches of the nodes.

The end result is that you will process the batch of prompts much faster if you reuse KV caches of shared prefixes across prompts, rather than processing each prompt independently.

The exact implementation details, e.g., the choice of data structure and the algorithms involved in managing the shared KV cache, are left to you. This is an open-ended design question.

### Hints

- **Attention with shared prefixes happens in several ways:**
  - When processing the shared prefix, you compute attention normally, with a triangular causal mask.
  - When processing attention for the distinct suffix, you must use a suitably shaped *rectangular* causal mask, while still ensuring that you attend only to past tokens.
  - When generating output tokens one by one, you compute only one row of the attention matrix, and you need not use an attention mask.
- **Position embeddings:** Every token undergoes a position embedding, by passing an array of position indices to the position embedding layer. When processing the entire sequence in the default code, token positions are simply passed as `0 ... N-1`, where `N` is the sequence length. Now, with the KV cache being reused, you must pass the correct position index values to the embedding layer, e.g., `P ... P+S-1`, where `P` is the size of the shared prefix (for which you need not compute anything, as the KV cache is available), and `S` is the distinct suffix length.
