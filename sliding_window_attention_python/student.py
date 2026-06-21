import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feel free to import any other library if needed


def set_seed(seed=42):
    """
    Makes everything deterministic for reproducible results.
    DO NOT MODIFY THIS FUNCTION.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SlidingWindowCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, window_size, seed=42):
        """
        Initializes the Sliding Window Causal Self-Attention layer.

        DO NOT CHANGE ANYTHING IN THIS FUNCTION.

        Args:
            n_embd: Embedding dimension (total across all heads)
            n_head: Number of attention heads
            window_size: Maximum number of previous tokens each token can attend to
            seed: Random seed for deterministic weight initialization
        """
        super().__init__()
        assert n_embd % n_head == 0

        set_seed(seed)

        self.n_embd = n_embd  # Embedding dimension (C)
        self.n_head = n_head  # Number of heads (H)
        self.window_size = window_size  # Window size (W)
        self.head_dim = n_embd // n_head  # Dimension per head (D)

        # Single linear layer that projects input to Q, K, V concatenated
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)

    def forward(self, x):
        """
        Forward pass of the Sliding Window Causal Self-Attention layer.

        Args:
            x: Input tensor of shape (B, T, C) where
               B = batch size
               T = sequence length
               C = embedding dimension (same as n_embd)

        Returns:
            y: Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()
        device = x.device

        # TODO: Extract q, k, v (in this order) using self.c_attn
        # Split the output along the correct dimension to get three separate tensors

        # TODO: Reshape q, k, v to separate into multiple heads

        # TODO: Compute attention scores using q and k
        # Remember to apply scaling

        # TODO: Create a sliding window causal mask
        # The mask should enforce: (1) causality - no attending to future tokens
        #                          (2) window constraint - only attend to last W tokens

        # TODO: Apply the mask to the attention scores

        # TODO: Apply softmax to obtain attention weights

        # TODO: Apply attention weights to values to get the output

        # TODO: Reshape the output back to (B, T, C)

        # TODO: Return the final output

        pass  # Remove this line when you implement the function


# TESTING CODE (YOU CAN USE THIS TO DEBUG YOUR IMPLEMENTATION)
if __name__ == "__main__":
    print("Testing your implementation...")

    set_seed(42)
    model = SlidingWindowCausalSelfAttention(
        n_embd=64, n_head=4, window_size=3, seed=42
    )
    model.eval()

    set_seed(42)
    x = torch.randn(2, 8, 64)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    print("\nIf you see this output, your code at least runs!")
    print("Use the provided grader script to verify correctness.")
