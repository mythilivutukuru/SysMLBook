#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 2D Convolution Kernel
 * 
 * Performs 2D convolution on an input matrix using a square kernel.
 * Each thread computes one output element by sliding the kernel over the input.
 * 
 * @param input    [IN]  Pointer to input matrix in device memory
 *                       Shape: [height x width]
 *                       Flattened row-major order
 * 
 * @param kernel   [IN]  Pointer to convolution kernel in device memory
 *                       Shape: [kernel_size x kernel_size]
 *                       Flattened row-major order
 * 
 * @param output   [OUT] Pointer to output matrix in device memory
 *                       Shape: [out_height x out_width] where
 *                       out_height = height - kernel_size + 1
 *                       out_width = width - kernel_size + 1
 *                       Flattened row-major order
 * 
 * @param height        [IN]  Height of input matrix (number of rows)
 * @param width         [IN]  Width of input matrix (number of columns)
 * @param kernel_size   [IN]  Size of square kernel (e.g., 3 for 3x3 kernel)
 * 
 * @return void
 * 
 * @note Thread organization: 2D grid of 2D blocks
 * @note Each thread computes: output[row][col] = sum(input_region * kernel)
 */
__global__ void conv2d_kernel(
    const float* input,  
    const float* kernel, 
    float* output,       
    int height,
    int width,
    int kernel_size
) {
    int out_width = width - kernel_size + 1;
    int out_height = height - kernel_size + 1;
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < out_height && col < out_width) {
        float sum = 0.0f;
        
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int input_row = row + i;
                int input_col = col + j;
                sum += input[input_row * width + input_col] * 
                       kernel[i * kernel_size + j];
            }
        }
        
        output[row * out_width + col] = sum;
    }
}

/**
 * @brief ReLU (Rectified Linear Unit) Activation Kernel
 * 
 * Applies ReLU activation function element-wise to input data.
 * ReLU(x) = max(0, x) - zeros out all negative values.
 * This is an in-place operation that modifies the input array.
 * 
 * @param data  [IN/OUT] Pointer to data array in device memory
 *                       Modified in-place: data[i] = max(0, data[i])
 *                       Shape: [size]
 *                       1D flattened array
 * 
 * @param size  [IN]     Total number of elements in the data array
 * 
 * @return void
 * 
 * @note In-place operation - input array is modified directly
 * @note Thread organization: 1D grid of 1D blocks
 * @note Each thread processes one element: data[idx] = max(0, data[idx])
 */
__global__ void relu_kernel(
    float* data, 
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

/* ============================================================
   TESTING INFRASTRUCTURE — students do not modify below
   ============================================================ */

// Tolerance for floating point comparison
#define EPSILON 1e-4

/**
 * @brief Compare two float arrays with tolerance
 */
bool compare_arrays(const float* actual, const float* expected, int size, const char* name) {
    bool passed = true;
    for (int i = 0; i < size; i++) {
        if (fabs(actual[i] - expected[i]) > EPSILON) {
            printf("    Mismatch at index %d: expected %.2f, got %.2f\n", 
                   i, expected[i], actual[i]);
            passed = false;
        }
    }
    if (passed) {
        printf("    %s output matches expected values\n", name);
    }
    return passed;
}

/**
 * @brief Run convolution + ReLU pipeline and verify results
 */
bool run_test_case(
    const char* test_name,
    float* h_input, int height, int width,
    float* h_kernel, int kernel_size,
    float* expected_conv, float* expected_relu
) {
    printf("\n========================================\n");
    printf("TEST: %s\n", test_name);
    printf("========================================\n");
    
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;
    int output_size = out_height * out_width;
    
    // Device pointers
    float *d_input, *d_kernel, *d_output;
    
    // Allocate device memory
    cudaMalloc(&d_input, height * width * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run Convolution
    dim3 blockSize(16, 16);
    dim3 gridSize((out_width + blockSize.x - 1) / blockSize.x,
                  (out_height + blockSize.y - 1) / blockSize.y);
    
    conv2d_kernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, height, width, kernel_size);
    cudaDeviceSynchronize();
    
    // Get convolution result
    float* h_conv_output = (float*)malloc(output_size * sizeof(float));
    cudaMemcpy(h_conv_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Run ReLU
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, output_size);
    cudaDeviceSynchronize();
    
    // Get final result
    float* h_relu_output = (float*)malloc(output_size * sizeof(float));
    cudaMemcpy(h_relu_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    printf("\nVerifying Convolution Output:\n");
    bool conv_passed = compare_arrays(h_conv_output, expected_conv, output_size, "Convolution");
    
    printf("\nVerifying ReLU Output:\n");
    bool relu_passed = compare_arrays(h_relu_output, expected_relu, output_size, "ReLU");
    
    bool overall_passed = conv_passed && relu_passed;
    
    if (overall_passed) {
        printf("\n  TEST PASSED: %s\n", test_name);
    } else {
        printf("\n  TEST FAILED: %s\n", test_name);
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_conv_output);
    free(h_relu_output);
    
    return overall_passed;
}

int main() {
    printf("    Testing convolution and ReLU activation kernels     \n");
    
    int tests_passed = 0;
    int total_tests = 3;
    
    // ============================================================
    // TEST CASE 1
    // ============================================================
    {
        float h_input[25] = {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25
        };
        
        float h_kernel[9] = {
            -1, -1, -1,
            -1,  8, -1,
            -1, -1, -1
        };
        
        // Expected convolution output (3x3)
        // Computed manually for position [0,0]:
        // sum = 1*(-1) + 2*(-1) + 3*(-1) + 6*(-1) + 7*8 + 8*(-1) + 11*(-1) + 12*(-1) + 13*(-1)
        //     = -1 - 2 - 3 - 6 + 56 - 8 - 11 - 12 - 13 = 0
        float expected_conv[9] = {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f
        };
        
        // ReLU output (all zeros remain zeros)
        float expected_relu[9] = {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f
        };
        
        if (run_test_case("1", 
                          h_input, 5, 5, h_kernel, 3, 
                          expected_conv, expected_relu)) {
            tests_passed++;
        }
    }
    
    // ============================================================
    // TEST CASE 2
    // ============================================================
    {
        float h_input[16] = {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
           13, 14, 15, 16
        };
        
        // Identity kernel (center = 1, rest = 0)
        float h_kernel[9] = {
            0, 0, 0,
            0, 1, 0,
            0, 0, 0
        };
        
        // Convolution output: extracts center value of each 3x3 region
        float expected_conv[4] = {
            6.0f, 7.0f,
           10.0f, 11.0f
        };
        
        // ReLU output (all positive, so unchanged)
        float expected_relu[4] = {
            6.0f, 7.0f,
           10.0f, 11.0f
        };
        
        if (run_test_case("2", 
                          h_input, 4, 4, h_kernel, 3, 
                          expected_conv, expected_relu)) {
            tests_passed++;
        }
    }
    
    // ============================================================
    // TEST CASE 3
    // ============================================================
    {
        float h_input[16] = {
            10,  5,  2,  1,
             8,  4,  1,  0,
             6,  3,  0, -1,
             4,  2, -1, -2
        };
        
        // Inverted kernel (creates negative outputs)
        float h_kernel[9] = {
            0, 0, 0,
            0, -2, 0,
            0, 0, 0
        };
        
        // Convolution output: -2 * center value
        float expected_conv[4] = {
            -8.0f, -2.0f,
            -6.0f,  0.0f
        };
        
        // ReLU output: max(0, x) - negatives become 0
        float expected_relu[4] = {
            0.0f, 0.0f,
            0.0f, 0.0f
        };
        
        if (run_test_case("3", 
                          h_input, 4, 4, h_kernel, 3, 
                          expected_conv, expected_relu)) {
            tests_passed++;
        }
    }
    
    // ============================================================
    // Final Summary
    // ============================================================
    printf("\n");
    printf("                    TEST SUMMARY                        \n");
    printf("  Tests Passed: %d / %d                                  \n", tests_passed, total_tests);
}
