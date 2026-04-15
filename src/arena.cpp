#include "arena.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// tensor arena impl

TensorArena::TensorArena(size_t size_in_bytes)
    : capacity(size_in_bytes), offset(0) {

  if (posix_memalign(reinterpret_cast<void **>(&memory), 64, size_in_bytes) !=
      0) {
    memory = nullptr;
  }

  if (!memory)
    throw std::bad_alloc();
}

TensorArena::~TensorArena() { std::free(memory); }

void *TensorArena::allocate(size_t bytes, size_t alignment) {
  uintptr_t current_ptr = reinterpret_cast<uintptr_t>(memory + offset);
  size_t padding = (alignment - (current_ptr % alignment)) % alignment;

  if (offset + padding + bytes > capacity) {
    return nullptr; // out of memory
  }

  offset += padding;
  void *ptr = memory + offset;
  offset += bytes;
  return ptr;
}

void TensorArena::reset() { offset = 0; }

size_t TensorArena::get_usage() const { return offset; }
size_t TensorArena::get_capacity() const { return capacity; }

// CUDA arena impl

CudaArena::CudaArena(size_t size_bytes) : total_size(size_bytes), offset(0) {
  cudaError_t err = cudaMalloc((void **)&d_base_ptr, total_size);
  if (err != cudaSuccess) {
    throw std::runtime_error("CudaArena failed to allocate VRAM");
  }
}

CudaArena::~CudaArena() { cudaFree(d_base_ptr); }

float *CudaArena::allocate(size_t bytes) {
  size_t aligned_bytes = (bytes + 255) & ~255;

  if (offset + aligned_bytes > total_size) {
    throw std::bad_alloc(); // out of VRAM
  }

  float *ptr = d_base_ptr + (offset / sizeof(float));
  offset += aligned_bytes;
  return ptr;
}

void CudaArena::reset() { offset = 0; }
