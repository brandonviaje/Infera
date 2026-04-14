#include "tensor.hpp"

Tensor::Tensor(TensorArena &arena, const std::vector<int> &tensor_shape,
               Device dev)
    : shape(tensor_shape), device(dev) {

  calculate_strides();

  size_t total_bytes = num_elements() * sizeof(float);
  raw_data = arena.allocate(total_bytes);

  if (!raw_data) {
    throw std::runtime_error("Tensor allocation failed: Arena out of memory.");
  }
}

void Tensor::calculate_strides() {
  strides.resize(shape.size());
  int current_stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = current_stride;
    current_stride *= shape[i];
  }
}

size_t Tensor::num_elements() const {
  if (shape.empty())
    return 0;
  size_t total = 1;
  for (int dim : shape)
    total *= dim;
  return total;
}

size_t Tensor::size_bytes() const { return num_elements() * sizeof(float); }
