#include "arena.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include <cuda_runtime.h>
#include <iostream>

int main() {
  try {
    std::cout << "Waking up GPU...\n";
    float *dummy;
    cudaMalloc(&dummy, sizeof(float));
    cudaFree(dummy);

    TensorArena arena(128 * 1024 * 1024); // allocate big ass arena

    std::vector<int> shape = {1024, 1024};

    Tensor Input(arena, shape);
    Tensor Weights(arena, shape);
    Tensor Bias(arena, shape);

    Tensor MatMul_Result(arena, shape);
    Tensor Add_Result(arena, shape);
    Tensor Final_Output(arena, shape);

    // init with test data
    float *in_data = Input.data_as<float>();
    float *w_data = Weights.data_as<float>();
    float *b_data = Bias.data_as<float>();

    for (size_t i = 0; i < Input.num_elements(); ++i) {
      in_data[i] = 1.0f;
      w_data[i] = 1.0f;
      b_data[i] = -1500.0f;
    }

    std::cout << "Executing Layer Forward Pass on GPU...\n";

    // matmul
    ops::matmul_cuda(Input, Weights, MatMul_Result);

    // add bias
    ops::add_cuda(MatMul_Result, Bias, Add_Result);

    // activation
    ops::relu_cuda(Add_Result, Final_Output);

    std::cout << "Layer Complete. Checking math...\n";

    // verify the math for first element

    // MatMul: 1024 elements of (1.0 * 1.0) added together = 1024.0
    // add bias: 1024.0 + (-1500.0) = -476.0
    // ReLU: max(0, -476.0) = 0.0

    float first_val = Final_Output.data_as<float>()[0];
    std::cout << "Expected first value: 0\n";
    std::cout << "Actual first value:   " << first_val << "\n";

    if (first_val == 0.0f) {
      std::cout << "SUCCESS: Engine successfully executed a layer!\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Fatal Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
