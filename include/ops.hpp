#ifndef OPS_HPP
#define OPS_HPP

#include "tensor.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace ops {

/**
 * Validates that dimensions match for matrix multiplication.
 * Throws std::invalid_argument if shapes are incompatible.
 */
void check_shapes(const Tensor &A, const Tensor &B, const Tensor &C);

/**
 * Baseline Matrix Multiplication (O(N^3)).
 * Standard triple-nested loop (I-J-K).
 */
void matmul_naive(Tensor &A, Tensor &B, Tensor &C);

/**
 * Optimized Matrix Multiplication.
 * Utilizes Loop Reordering (I-K-J) for linear memory access and
 * Cache Tiling (Blocking) to maximize L1/L2 cache hits.
 */
void matmul_optimized(Tensor &A, Tensor &B, Tensor &C);

/**
 * GPU-Accelerated Matrix Multiplication using CUDA.
 * Offloads computation to the GPU for massive parallelism.
 * Assumes input tensors are in row-major layout and compatible in shape.
 * Includes device memory allocation, data transfer, kernel launch, and synchronization.
 */
void matmul_cuda(Tensor& A, Tensor& B, Tensor& C);

/**
 * Tensor Addition (CPU).
 * Computes C = A + B using a simple loop over all elements
 * Assumes all tensors have identical shapes.
 */
void add_cpu(Tensor& A, Tensor& B, Tensor& C);

/**
 * Tensor Addition (CUDA).
 * Performs parallel addition on GPU
 * Each thread handles one or more elements depending on tensor size.
 */
void add_cuda(Tensor& A, Tensor& B, Tensor& C);

/**
 * ReLU Activation (CPU).
 * Applies Y = max(0, X) element-wise.
 */
void relu_cpu(Tensor& X, Tensor& Y);

/**
 * ReLU Activation (CUDA).
 * Parallel GPU implementation of ReLU.
 */
void relu_cuda(Tensor& X, Tensor& Y);

} 

#endif
