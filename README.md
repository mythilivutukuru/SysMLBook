# SysML Book – Programming Assignments

Companion repository for the textbook **"Systems for Machine Learning"**.

📖 [**Textbook**](https://www.cse.iitb.ac.in/~mythili/sysml/)

Each assignment gives you hands-on practice with the concepts covered in the book. Work through them chapter by chapter alongside your reading.

---

## How This Repo Is Organized

Each assignment lives in its own folder with these files:

| File | What it contains |
|------|-----------------|
| `README.md` | Problem statement, background, hints, and how to check your solution |
| `student.py` / `student.cu` / `student.cpp` | Starter code with `TODO` sections for you to fill in |
| `solution.py` / `solution.cu` / `solution.cpp` | Complete solution *(provided for selected assignments only)* |

---

## Getting Started

1. Clone the repo:
```bash
   git clone https://github.com/mythilivutukuru/SysMLBook
   cd SysMLBook
```
2. Open the folder for the assignment you want to work on.
3. Read that folder's `README.md` first — it explains the problem and gives hints.
4. Edit `student.py` (or `student.cu`/ `student.cpp`) and fill in the `TODO` sections.
5. Follow the evaluation instructions in the folder's `README.md` to test your work.

---

## Assignments

### Chapter 3: Programming AI Hardware

| Assignment | Topic | Language |
|------------|-------|----------|
| [SIMD](./simd) | SIMD Matrix Multiplication | C++ |
| [Vector_Addition](./vector_addition/) | Vector Addition | CUDA |
| [CNN](./convolutional_neural_network/) | Convolutional Neural Network | CUDA |
| [Sliding_Window_Attention](./sliding_window_attention_cuda/) | Sliding Window Attention | CUDA |
| [Matrix_Multiplication](./matrix_multiplication/) | Matrix Multiplication | CUDA |
| [Matrix_Transpose](./matrix_transpose_shared_memory/) | Matrix Transpose Using Shared Memory | CUDA |
| [Standard_Attention](./standard_attention_shared_memory/) | Standard Attention Using Shared Memory | CUDA |
| [MLP_Forward_Pass](./mlp_forward_pass/) | MLP Forward Pass | CUDA |
| [MLP_Forward_Pass_Streams](./mlp_forward_pass_streams/) | MLP Forward Pass with Streams | CUDA |
| [Matrix_Multiplication_Tensor_Cores](./matrix_multiplication_tensor_cores/) | Matrix Multiplication Using Tensor Cores | CUDA |

> More chapters and assignments will be added over time.

---

## Citation

If you use this material, please cite the textbook:

```bibtex
@book{vutukuru_sysml,
  title       = {Systems for Machine Learning},
  author      = {Mythili Vutukuru and Saksham Rathi},
  year        = {2026},
  institution = {Indian Institute of Technology Bombay},
  url         = {https://www.cse.iitb.ac.in/~mythili/sysml/}
}
```

---