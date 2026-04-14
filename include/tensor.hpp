#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "arena.hpp"
#include <numeric>
#include <vector>

enum class Device { CPU, GPU };

class Tensor {
public:
  /**
   * @param arena The memory pool to pull from.
   * @param shape The dimensions (e.g., {1, 3, 224, 224}).
   * @param dev Physical location of the data.
   */
  Tensor(TensorArena &arena, const std::vector<int> &shape,
         Device dev = Device::CPU);

  // Getters
  size_t num_elements() const;
  size_t size_bytes() const;

  // Low-level data access
  template <typename T = float> T *data_as() {
    return reinterpret_cast<T *>(raw_data);
  }

  const std::vector<int> &get_shape() const { return shape; }
  const std::vector<int> &get_strides() const { return strides; }

private:
  void *raw_data;
  std::vector<int> shape;
  std::vector<int> strides;
  Device device;

  void calculate_strides();
};

#endif
