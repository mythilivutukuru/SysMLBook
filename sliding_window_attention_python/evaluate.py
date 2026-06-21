# DO NOT MODIFY ANYTHING IN THIS FILE!!!

import torch
import torch.nn.functional as F
import json
from student import SlidingWindowCausalSelfAttention, set_seed


def load_and_grade(student_model_class, json_file="client_test_cases.json", tolerance=1e-5):
    """
    Load test cases and grade student implementation
    
    Args:
        student_model_class: The student's SlidingWindowCausalSelfAttention class
        json_file: Path to test cases JSON
        tolerance: Numerical tolerance for floating point comparison
    """
    # Load test cases
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    test_cases = data["test_cases"]
    passed = 0
    failed = 0
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    for test in test_cases:
        test_id = test["test_id"]
        params = test["parameters"]
        expected = test["expected_output"]
        
        try:
            # Set seed
            set_seed(params["seed"])
            
            # Create student model
            model = student_model_class(
                n_embd=params["embedding_dim"],
                n_head=params["num_heads"],
                window_size=params["window_size"],
                seed=params["seed"]
            )
            model.eval()
            
            # Create input
            set_seed(params["seed"])
            x = torch.randn(
                params["batch_size"],
                params["sequence_length"],
                params["embedding_dim"]
            )
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            # Check shape
            if list(output.shape) != expected["shape"]:
                print(f"❌ Test {test_id} FAILED: Shape mismatch")
                print(f"   Expected: {expected['shape']}, Got: {list(output.shape)}")
                failed += 1
                continue
            
            # Check statistics
            mean_diff = abs(output.mean().item() - expected["mean"])
            std_diff = abs(output.std().item() - expected["std"])
            sum_diff = abs(output.sum().item() - expected["sum"])
            
            # Check first values
            first_vals = output.flatten()[:10].tolist()
            first_vals_match = all(
                abs(a - b) < tolerance 
                for a, b in zip(first_vals, expected["first_10_values"])
            )
            
            if (mean_diff < tolerance and 
                std_diff < tolerance and 
                sum_diff < tolerance * 10 and 
                first_vals_match):
                print(f"✅ Test {test_id} PASSED")
                passed += 1
            else:
                print(f"❌ Test {test_id} FAILED:")
                print(f"   Mean diff: {mean_diff:.2e} (threshold: {tolerance:.2e})")
                print(f"   Std diff: {std_diff:.2e}")
                print(f"   Sum diff: {sum_diff:.2e}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Test {test_id} FAILED with exception: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "="*50)
    print(f"RESULTS: {passed}/{len(test_cases)} tests passed")
    print(f"Score: {(passed/len(test_cases))*100:.1f}%")
    print("="*50)
    
    return passed, failed


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing reference implementation...")
    print("="*50)
    passed, failed = load_and_grade(SlidingWindowCausalSelfAttention, "client_test_cases.json")