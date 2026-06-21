import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=42):
    """Make everything deterministic"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SlidingWindowCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, window_size, seed=42):
        super().__init__()
        assert n_embd % n_head == 0

        set_seed(seed)

        self.n_embd = n_embd
        self.n_head = n_head
        self.window_size = window_size
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        D = self.head_dim
        W = self.window_size
        device = x.device

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))

        q_pos = torch.arange(T, device=device).unsqueeze(-1)
        k_pos = torch.arange(T, device=device).unsqueeze(0)

        causal = k_pos <= q_pos
        window = k_pos >= (q_pos - W + 1)
        mask = causal & window

        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return y
