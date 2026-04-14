#include "ops.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // calculate global row and column specific thread responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // bounds check
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum; // write dot product to memory
    }
}

__global__ void add_kernel(const float* A, const float* B, float* C, int num_elements) {
    // 1D grid calculation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void relu_kernel(const float* X, float* Y, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // ReLU: max(0, x)
        Y[idx] = X[idx] > 0.0f ? X[idx] : 0.0f;
    }
}

namespace ops {

void check_shapes(const Tensor &A, const Tensor &B, const Tensor &C) {
  if (A.get_shape().size() != 2 || B.get_shape().size() != 2 ||
      C.get_shape().size() != 2) {
    throw std::invalid_argument("MatMul currently only supports 2D tensors.");
  }
  if (A.get_shape()[1] != B.get_shape()[0]) {
    throw std::invalid_argument(
        "Inner dimensions must match for MatMul (M x K) * (K x N).");
  }
  if (C.get_shape()[0] != A.get_shape()[0] ||
      C.get_shape()[1] != B.get_shape()[1]) {
    throw std::invalid_argument("Output tensor shape is incorrect.");
  }
}

// naive matmul
void matmul_naive(Tensor &A, Tensor &B, Tensor &C) {
  check_shapes(A, B, C);

  int M = A.get_shape()[0];
  int K = A.get_shape()[1];
  int N = B.get_shape()[1];

  float *a_data = A.data_as<float>();
  float *b_data = B.data_as<float>();
  float *c_data = C.data_as<float>();

  // std I-J-K loop
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += a_data[i * K + k] * b_data[k * N + j];
      }
      c_data[i * N + j] = sum;
    }
  }
}

// loop reordering and tiling
void matmul_optimized(Tensor &A, Tensor &B, Tensor &C) {
  check_shapes(A, B, C);

  int M = A.get_shape()[0];
  int K = A.get_shape()[1];
  int N = B.get_shape()[1];

  float *a_data = A.data_as<float>();
  float *b_data = B.data_as<float>();
  float *c_data = C.data_as<float>();

  std::memset(c_data, 0, C.size_bytes());

  const int BLOCK_SIZE = 32;

  // tile the loops to keep chunks of A and B in cache
  for (int i_step = 0; i_step < M; i_step += BLOCK_SIZE) {
    for (int k_step = 0; k_step < K; k_step += BLOCK_SIZE) {
      for (int j_step = 0; j_step < N; j_step += BLOCK_SIZE) {

        // compute mini-matrix mul for this block
        for (int i = i_step; i < std::min(i_step + BLOCK_SIZE, M); ++i) {
          for (int k = k_step; k < std::min(k_step + BLOCK_SIZE, K); ++k) {

            float a_val = a_data[i * K + k];

            for (int j = j_step; j < std::min(j_step + BLOCK_SIZE, N); ++j) {
              c_data[i * N + j] += a_val * b_data[k * N + j];
            }
          }
        }
      }
    }
  }
}

void matmul_cuda(Tensor& A, Tensor& B, Tensor& C) {
    check_shapes(A, B, C);

    int M = A.get_shape()[0];
    int K = A.get_shape()[1];
    int N = B.get_shape()[1];

    std::size_t size_A = A.size_bytes();
    std::size_t size_B = B.size_bytes();
    std::size_t size_C = C.size_bytes();

    // alloc GPU VRAM
    float *d_A, *d_B, *d_C; 
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A.data_as<float>(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data_as<float>(), size_B, cudaMemcpyHostToDevice);

    // define grid and block dims
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();
    cudaMemcpy(C.data_as<float>(), d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void add_cpu(Tensor& A, Tensor& B, Tensor& C) {
    size_t num_elements = A.num_elements();
    float* a_data = A.data_as<float>();
    float* b_data = B.data_as<float>();
    float* c_data = C.data_as<float>();

    for (size_t i = 0; i < num_elements; ++i) {
        c_data[i] = a_data[i] + b_data[i];
    }
}

void relu_cpu(Tensor& X, Tensor& Y) {
    size_t num_elements = X.num_elements();
    float* x_data = X.data_as<float>();
    float* y_data = Y.data_as<float>();

    for (size_t i = 0; i < num_elements; ++i) {
        y_data[i] = x_data[i] > 0.0f ? x_data[i] : 0.0f;
    }
}

void add_cuda(Tensor& A, Tensor& B, Tensor& C) {
    int num_elements = A.num_elements();
    size_t size_bytes = A.size_bytes();

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_bytes);
    cudaMalloc(&d_B, size_bytes);
    cudaMalloc(&d_C, size_bytes);

    cudaMemcpy(d_A, A.data_as<float>(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data_as<float>(), size_bytes, cudaMemcpyHostToDevice);

    // 256 threads per block is the sweet spot for 1D arrays on NVIDIA cards
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data_as<float>(), d_C, size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void relu_cuda(Tensor& X, Tensor& Y) {
    int num_elements = X.num_elements();
    size_t size_bytes = X.size_bytes();

    float *d_X, *d_Y;
    cudaMalloc(&d_X, size_bytes);
    cudaMalloc(&d_Y, size_bytes);

    cudaMemcpy(d_X, X.data_as<float>(), size_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, num_elements);
    cudaDeviceSynchronize();

    cudaMemcpy(Y.data_as<float>(), d_Y, size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_Y);
}

} // namespace ops
