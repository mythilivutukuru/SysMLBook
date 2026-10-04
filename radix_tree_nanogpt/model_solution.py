"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# Radix Tree Related Classes
class RadixNode:
    __slots__ = ("tokens", "children", "kv")

    def __init__(self, tokens):
        self.tokens = tokens              # List[int]
        self.children = {}                # first_token -> RadixNode
        self.kv = None                    # KV slice exactly len(tokens)

def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class BatchRadixTree:
    def __init__(self):
        self.root = RadixNode([])

    def insert_sequence(self, tokens):
        node = self.root
        i = 0

        while i < len(tokens):
            t = tokens[i]

            # No child → create leaf
            if t not in node.children:
                node.children[t] = RadixNode(tokens[i:])
                return

            child = node.children[t]
            overlap = common_prefix_len(tokens[i:], child.tokens)

            # Full match → descend
            if overlap == len(child.tokens):
                node = child
                i += overlap
                continue

            # Partial match → Split
            split = RadixNode(child.tokens[:overlap])

            # Old remainder
            child.tokens = child.tokens[overlap:]
            split.children[child.tokens[0]] = child

            # New remainder
            remaining = tokens[i + overlap:]
            if remaining:
                split.children[remaining[0]] = RadixNode(remaining)

            node.children[t] = split
            return

    def find_cached_prefix(self, tokens):
        node = self.root
        i = 0
        matched = 0
        kvs = []

        while i < len(tokens):
            t = tokens[i]
            if t not in node.children:
                break

            child = node.children[t]
            overlap = common_prefix_len(tokens[i:], child.tokens)

            if overlap < len(child.tokens):
                break

            if child.kv is not None:
                kvs.append(child.kv)
                matched += overlap

            node = child
            i += overlap

        if not kvs:
            return None, 0

        kv = kvs[0]
        for k in kvs[1:]:
            kv = kv_append(kv, k)

        return kv, matched

    def store_kv_along_path(self, tokens, full_kv):
        node = self.root
        i = 0
        kv_pos = 0

        while i < len(tokens):
            child = node.children[tokens[i]]
            span = len(child.tokens)

            if child.kv is None:
                child.kv = kv_slice(full_kv, kv_pos, kv_pos + span)

            node = child
            i += span
            kv_pos += span

    def dump(self):
        def _dump(node, prefix):
            print(f"{prefix} Tokens: {node.tokens}, KV: {'Yes' if node.kv else 'No'}")
            for child in node.children.values():
                _dump(child, prefix + "  ")

        _dump(self.root, "")




def kv_append(kv_old, kv_new):
    """Append new KV to old KV along sequence dimension"""
    out = []
    for (k1, v1), (k2, v2) in zip(kv_old, kv_new):
        out.append((
            torch.cat((k1, k2), dim=1),
            torch.cat((v1, v2), dim=1)
        ))
    return tuple(out)


def kv_slice(kv, start, end):
    """Slice KV cache from start to end along sequence dimension"""
    out = []
    for k, v in kv:
        out.append((
            k[:, start:end, :].contiguous(),
            v[:, start:end, :].contiguous()
        ))
    return tuple(out)


def build_radix_tree_from_batch(token_sequences):
    tree = BatchRadixTree()
    
    for tokens in token_sequences:
        tree.insert_sequence(tokens)
    
    return tree




class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        if kv:
            prev_k, prev_v = kv
            k = torch.cat((prev_k, k), dim=1)
            v = torch.cat((prev_v, v), dim=1)
        
        new_kv = (k, v)
        curr_T = k.shape[1]

        k = k.view(B, curr_T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, curr_T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        elif kv and T == 1:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # token by token generation so masking not needed
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            S = k.size(-2) # S = current sequence length
            mask = self.bias[:, :, :T, :S] == 0
            att = att.masked_fill(mask, -float('inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv=None):
        x_attn, elem = self.attn(self.ln_1(x), kv)
        x = x + x_attn
        x = x + self.mlp(self.ln_2(x))
        return x, elem

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        if kv and kv[0] is not None:
            past_len = kv[0][0].shape[1]
        else:
            past_len = 0

        # adjust position embedding 
        pos = (torch.arange(past_len, past_len + t, device=idx.device) % self.config.block_size).unsqueeze(0)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        if not kv:
            kv = [None] * len(self.transformer.h)

        new_kv = []
        for block, kv_bl in zip(self.transformer.h, kv):
            x, elem = block(x, kv=kv_bl)
            new_kv.append(elem)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, new_kv

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        B, T = idx.shape
        assert B == 1, "Radix KV cache supports batch size = 1"

        tokens = idx[0].tolist()

        kv = None
        matched_len = 0


        if matched_len < len(tokens):
            prefill = idx[:, matched_len:] # we are passing only those tokens that were not cached, note that this impacts both positional and token embeddings
            _, _, new_kv = self(prefill, kv=kv)

            kv = new_kv if kv is None else kv_append(kv, new_kv) # append the newly computed KV to the reused prefix KV


        for _ in range(max_new_tokens):
            logits, _, new_kv = self(idx[:, -1:], kv=kv)  

            kv = kv_append(kv, new_kv)

            if kv[0][0].size(1) > self.config.block_size:
                kv = kv_slice(kv, -self.config.block_size, None)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)
            tokens.append(idx_next.item())

        return idx
    
    def generate_batch_with_radix(model, batch_prompts, max_new_tokens, temperature=1.0, top_k=None):
        token_sequences = [prompt[0].tolist() for prompt in batch_prompts]
        radix_tree = build_radix_tree_from_batch(token_sequences)

        radix_tree.dump()
        
        results = []
        
        for prompt_idx, (prompt_tokens, tokens) in enumerate(zip(batch_prompts, token_sequences)):
            print(f"\nProcessing request {prompt_idx + 1}/{len(batch_prompts)}")
            
            cached_kv, matched_len = radix_tree.find_cached_prefix(tokens)
            
            print(f"  Cached prefix length: {matched_len}/{len(tokens)}")
            
            kv = cached_kv
            if matched_len < len(tokens):
                uncached_tokens = prompt_tokens[:, matched_len:]
                print(f"  Computing KV for {uncached_tokens.shape[1]} uncached tokens")
                
                _, _, new_kv = model(uncached_tokens, kv=kv)
                
                kv = new_kv if kv is None else kv_append(kv, new_kv)
                
                radix_tree.store_kv_along_path(tokens, kv)
            
            # Step 2d: Generate new tokens
            idx = prompt_tokens.clone()
            generated_tokens = tokens.copy()
            
            for step in range(max_new_tokens):
                # Forward pass for single token
                logits, _, new_kv = model(idx[:, -1:], kv=kv)
                kv = kv_append(kv, new_kv)
                
                # Keep KV cache within block size
                if kv[0][0].size(1) > model.config.block_size:
                    kv = kv_slice(kv, -model.config.block_size, None)
                
                # Sample next token
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                idx = torch.cat([idx, idx_next], dim=1)
                generated_tokens.append(idx_next.item())
            
            results.append(idx)
        
        return results