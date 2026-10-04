"""
Evaluation script to compare KV caching vs no KV caching
Benchmarks generation speed, compares outputs, and verifies KV cache shapes
"""
from contextlib import nullcontext
import torch
import tiktoken
from model import GPT
import time


# Configuration
CONFIG = {
    'init_from': 'gpt2',
    'seed': 1337,
    'device': 'cpu',
    'dtype': 'float32',
    'compile': False,
    'num_samples': 3,
    'max_new_tokens': 10
}


def create_system_prompt():
    """Create a large system prompt for testing."""
    base_prompt = """HI!!! You are a helpful, harmless, and honest AI assistant.
Your purpose is to be helpful to users while being safe and truthful. You should:
- Provide accurate and factual information to the best of your knowledge
- Acknowledge when you're uncertain or don't know something
- Refuse harmful requests politely and explain why
- Communicate clearly and adjust to the user's level of expertise
- Think step-by-step for complex problems
- Ask clarifying questions when needed
Remember to always prioritize user safety and wellbeing in your responses.
You have broad knowledge spanning science, mathematics, history, arts, technology, and more.
You can help with analysis, writing, coding, math, creative tasks, and general questions.
Always strive to be helpful while maintaining ethical standards."""
    
    return base_prompt * 2


def prepare_sequences(system_prompt, user_prompts, encode, device):
    """Prepare full sequences from system prompt and user prompts."""
    system_ids = encode(system_prompt)
    
    full_sequences = []
    for user_prompt in user_prompts:
        user_ids = encode(" " + user_prompt)
        full_ids = system_ids + user_ids
        full_sequences.append((torch.tensor(full_ids, dtype=torch.long, device=device)[None, ...]))
    
    return full_sequences, system_ids


def verify_kv_shapes(prefix_kvs, model, expected_prefix_len):
    """Verify the shapes of the prefix KV cache."""
    print(f"\n{'='*80}")
    print("KV CACHE SHAPE VERIFICATION")
    print(f"{'='*80}")
    
    if prefix_kvs is None:
        print("⚠ WARNING: prefix_kvs is None")
        print("This means no prefix sharing occurred")
        return False
    
    all_valid = True

    print(f"Number of layers: {len(prefix_kvs)}")
    print(f"Expected prefix length: {expected_prefix_len}")
    
    if len(prefix_kvs) != 12:
        print(f"  ✗ ERROR: Number of layers not as expected")
        all_valid = False
    
    for layer_idx, (k, v) in enumerate(prefix_kvs):
        # Expected shape: (batch_size=1, n_head, seq_len, head_dim)
        k_shape = k.shape
        v_shape = v.shape
        
        # Verify shapes match
        if k_shape != v_shape:
            print(f"  ✗ ERROR: Key and Value shapes don't match!")
            all_valid = False
        
        # Verify batch size is 1
        if k_shape[0] != 1:
            print(f"  ✗ ERROR: Batch size should be 1, got {k_shape[0]}")
            all_valid = False
        
        # Verify sequence length matches expected prefix length
        seq_len = k_shape[2]
        if seq_len != expected_prefix_len:
            print(f"  ⚠ WARNING: Sequence length {seq_len} != expected {expected_prefix_len}")
        
        # Verify number of heads
        n_head = k_shape[1]
        expected_n_head = model.config.n_head
        if n_head != expected_n_head:
            print(f"  ✗ ERROR: Number of heads {n_head} != expected {expected_n_head}")
            all_valid = False
        
        # Verify head dimension
        head_dim = k_shape[3]
        expected_head_dim = model.config.n_embd // model.config.n_head
        if head_dim != expected_head_dim:
            print(f"  ✗ ERROR: Head dimension {head_dim} != expected {expected_head_dim}")
            all_valid = False
    
    print(f"\n{'='*80}")
    print(f"Overall KV cache validity: {'✓ VALID' if all_valid else '✗ INVALID'}")
    print(f"{'='*80}\n")
    
    return all_valid


def benchmark_generation(model, sequences, max_new_tokens, enable_prefix_sharing, num_samples):
    """Run generation benchmark and return times, outputs, and prefix KVs."""
    times = []
    all_outputs = []
    prefix_kvs = None
    
    for k in range(num_samples):
        start_time = time.time()
        with torch.no_grad():
            outputs, shared_prefix_kvs = model.generate(
                sequences, 
                max_new_tokens, 
                enable_prefix_sharing=enable_prefix_sharing
            )
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Store outputs and prefix KVs from first run
        if k == 0:
            all_outputs = outputs
            prefix_kvs = shared_prefix_kvs
    
    return times, all_outputs, prefix_kvs


def compare_outputs(outputs_without, outputs_with, decode):
    print(f"\n{'='*80}")
    print("OUTPUT COMPARISON")
    print(f"{'='*80}")

    # Normalize to lists if single tensor returned
    if isinstance(outputs_without, torch.Tensor):
        outputs_without = [outputs_without]
    if isinstance(outputs_with, torch.Tensor):
        outputs_with = [outputs_with]

    print(f"Sequences without prefix sharing: {len(outputs_without)}")
    print(f"Sequences with prefix sharing:    {len(outputs_with)}")

    if len(outputs_without) == 0 or len(outputs_with) == 0:
        print("✗ ERROR: One or both output lists are empty — cannot compare.")
        return False

    all_match = True
    for i, (out_without, out_with) in enumerate(zip(outputs_without, outputs_with)):
        match = (out_without == out_with).all().item()
        all_match = all_match and match

        print(f"\nSequence {i+1}:")
        print(f"  Without prefix sharing: {decode(out_without[0].tolist())!r}")
        print(f"  With prefix sharing:    {decode(out_with[0].tolist())!r}")
        print(f"  Match: {'✓ YES' if match else '✗ NO'}")
        if not match:
            print(f"  Token differences: {(out_without != out_with).sum().item()}")

    if len(outputs_without) != len(outputs_with):
        print(f"\n✗ WARNING: Output count mismatch ({len(outputs_without)} vs {len(outputs_with)})")
        all_match = False

    print(f"\n{'='*80}")
    print(f"All outputs match: {'✓ YES' if all_match else '✗ NO'}")
    print(f"{'='*80}\n")

    return all_match


def print_summary(times_without, times_with, avg_without, avg_with):
    """Print benchmark summary statistics."""
    speedup = avg_without / avg_with
    time_saved = avg_without - avg_with
    percent_saved = (1 - avg_with / avg_without) * 100
    
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Without prefix sharing:")
    print(f"  Average time: {avg_without:.3f}s")
    print(f"  Individual runs: {', '.join(f'{t:.3f}s' for t in times_without)}")
    print(f"\nWith prefix sharing:")
    print(f"  Average time: {avg_with:.3f}s")
    print(f"  Individual runs: {', '.join(f'{t:.3f}s' for t in times_with)}")
    print(f"\nImprovement:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.3f}s ({percent_saved:.1f}%)")
    print(f"{'='*80}\n")


def benchmark_prefix_sharing(model, encode, decode, device, num_samples=3, max_new_tokens=50):
    """
    Benchmark generation with and without prefix sharing.
    Compares performance, output correctness, and KV cache shapes.
    """
    
    # Setup prompts
    system_prompt = create_system_prompt()
    user_prompts = [
        "Once in a blue moon,",
        "The quick brown fox and fox,",
        "In a galaxy far, far, far away",
        "To be or not to be great,",
    ]
    
    # Prepare sequences
    full_sequences, system_ids = prepare_sequences(system_prompt, user_prompts, encode, device)
    prefix_len = len(system_ids)
    
    # Print configuration
    print(f"\n{'='*80}")
    print(f"BENCHMARKING PREFIX SHARING")
    print(f"{'='*80}")
    print(f"Number of sequences: {len(full_sequences)}")
    print(f"System prompt tokens: {prefix_len}")
    print(f"Tokens to generate: {max_new_tokens}")
    print(f"Number of runs: {num_samples}")
    
    # Benchmark WITHOUT prefix sharing
    print(f"\n{'-'*80}")
    print("WITHOUT PREFIX SHARING")
    print(f"{'-'*80}")
    
    times_without, outputs_without, prefix_kvs_without = benchmark_generation(
        model, full_sequences, max_new_tokens, False, num_samples
    )
    
    for k, t in enumerate(times_without, 1):
        print(f"Run {k}: {t:.3f}s")
    
    avg_time_without = sum(times_without) / len(times_without)
    
    # Verify that no prefix KVs were created when disabled
    print(f"\nPrefix KVs created: {'✗ YES (UNEXPECTED!)' if prefix_kvs_without is not None else '✓ NO (Expected)'}")
    
    # Benchmark WITH prefix sharing
    print(f"\n{'-'*80}")
    print("WITH PREFIX SHARING")
    print(f"{'-'*80}")
    
    times_with, outputs_with, prefix_kvs_with = benchmark_generation(
        model, full_sequences, max_new_tokens, True, num_samples
    )
    
    for k, t in enumerate(times_with, 1):
        print(f"Run {k}: {t:.3f}s")
    
    avg_time_with = sum(times_with) / len(times_with)
    
    # Verify prefix KVs shapes
    expected_prefix_len = min(prefix_len, model.config.block_size)
    kv_valid = verify_kv_shapes(prefix_kvs_with, model, expected_prefix_len)
    
    # Compare outputs
    outputs_match = compare_outputs(outputs_without, outputs_with, decode)
    
    # Print summary
    print_summary(times_without, times_with, avg_time_without, avg_time_with)
    
    return {
        'times_without': times_without,
        'times_with': times_with,
        'avg_without': avg_time_without,
        'avg_with': avg_time_with,
        'speedup': avg_time_without / avg_time_with,
        'outputs_match': outputs_match,
        'kv_shapes_valid': kv_valid,
        'prefix_kvs': prefix_kvs_with
    }


def setup_model(config):
    """Initialize model and environment."""
    # Set random seeds
    torch.manual_seed(config['seed'])
    if 'cuda' in config['device']:
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Setup device and dtype
    device_type = 'cuda' if 'cuda' in config['device'] else 'cpu'
    ptdtype = {
        'float32': torch.float32, 
        'bfloat16': torch.bfloat16, 
        'float16': torch.float16
    }[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype
    )
    
    # Load model
    print(f"Loading model: {config['init_from']}")
    model = GPT.from_pretrained(config['init_from'], dict(dropout=0.0))
    model.eval()
    model.to(config['device'])
    
    if config['compile']:
        print("Compiling model...")
        model = torch.compile(model)
    
    return model, ctx


def setup_tokenizer():
    """Setup GPT-2 tokenizer."""
    print("Setting up GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


def main():
    """Main execution function."""
    # Setup
    model, ctx = setup_model(CONFIG)
    encode, decode = setup_tokenizer()
    
    # Run benchmark
    results = benchmark_prefix_sharing(
        model=model,
        encode=encode,
        decode=decode, 
        device=CONFIG['device'],
        num_samples=CONFIG['num_samples'],
        max_new_tokens=CONFIG['max_new_tokens']
    )
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"✓ Outputs match: {results['outputs_match']}")
    print(f"✓ KV cache shapes valid: {results['kv_shapes_valid']}")
    print(f"✓ Speedup achieved: {results['speedup']:.2f}x")
    print("="*80 + "\n")
    
    return results


if __name__ == '__main__':
    main()