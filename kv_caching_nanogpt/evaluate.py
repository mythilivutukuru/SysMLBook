"""
Evaluation script to compare KV caching vs no KV caching
"""
from contextlib import nullcontext
import torch
import tiktoken
from model import GPT
import time

test_cases = [
    {"prompt": "Once upon a time", "max_tokens": 10},
    {"prompt": "The quick brown fox", "max_tokens": 50},
    {"prompt": "In a galaxy far, far away", "max_tokens": 100},
    {"prompt": "To be or not to be", "max_tokens": 500},
]

init_from = 'gpt2' 
seed = 1337
device = 'cpu' 
dtype = 'float32'
compile = False

# -----------------------------------------------------------------------------

def evaluate_generation(model, prompt, max_new_tokens, encode, decode, device, ctx):
    """
    Evaluate generation with and without KV caching.
    Returns metrics for both approaches.
    """
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # ===== WITHOUT KV CACHING =====
    torch.manual_seed(seed) 
    
    with torch.no_grad():
        with ctx:
            start_time = time.perf_counter()
            output_no_cache = model.generate(x, max_new_tokens, enable_kv_caching=False)
            end_time = time.perf_counter()
    
    time_no_cache = end_time - start_time
    output_text_no_cache = decode(output_no_cache[0].tolist())
    
    # ===== WITH KV CACHING =====
    torch.manual_seed(seed) 
    
    with torch.no_grad():
        with ctx:
            start_time = time.perf_counter()
            output_cache = model.generate(x, max_new_tokens, enable_kv_caching=True)
            end_time = time.perf_counter()
    
    time_cache = end_time - start_time
    output_text_cache = decode(output_cache[0].tolist())
    
    # ===== METRICS =====
    outputs_match = torch.equal(output_no_cache, output_cache)
    speedup = time_no_cache / time_cache if time_cache > 0 else float('inf')
    
    return {
        'outputs_match': outputs_match,
        'time_no_cache': time_no_cache,
        'time_cache': time_cache,
        'speedup': speedup,
        'output_text_no_cache': output_text_no_cache,
        'output_text_cache': output_text_cache,
    }

# -----------------------------------------------------------------------------
# Setup
torch.manual_seed(seed)
if 'cuda' in device:
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
print(f"Loading model: {init_from}")
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.eval()
model.to(device)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

# Setup encoder/decoder
print("Setting up GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# -----------------------------------------------------------------------------
# Run evaluations
print("\n" + "="*80)
print("KV CACHING EVALUATION - 4 TEST CASES")
print("="*80)

all_results = []

for idx, test_case in enumerate(test_cases, 1):
    prompt = test_case["prompt"]
    max_tokens = test_case["max_tokens"]
    
    print(f"\n{'='*80}")
    print(f"Test Case {idx}")
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_tokens}")
    print(f"{'='*80}")
    
    results = evaluate_generation(
        model, prompt, max_tokens, encode, decode, device, ctx
    )
    
    all_results.append({
        'prompt': prompt,
        'max_tokens': max_tokens,
        **results
    })
    
    # Print results
    print(f"\n✓ Outputs Match: {results['outputs_match']}")
    print(f"\nTiming:")
    print(f"  Without KV Cache: {results['time_no_cache']:.4f}s")
    print(f"  With KV Cache:    {results['time_cache']:.4f}s")
    print(f"  Speedup:          {results['speedup']:.2f}x")
    
    if not results['outputs_match']:
        print(f"\n⚠️  WARNING: Outputs don't match!")
        print(f"  No Cache: {results['output_text_no_cache'][:100]}...")
        print(f"  Cache:    {results['output_text_cache'][:100]}...")

print("\n" + "="*80)