#include "arena.hpp"
#include <iostream>

#ifdef _WIN32
#include <malloc.h>
#endif

TensorArena::TensorArena(size_t size_in_bytes)
    : capacity(size_in_bytes), offset(0) {

  if (posix_memalign(reinterpret_cast<void **>(&memory), 64, size_in_bytes) !=
      0) {
    memory = nullptr;
  }

  if (!memory)
    throw std::bad_alloc();
}

TensorArena::~TensorArena() {
#ifdef _WIN32
  _aligned_free(memory);
#else
  std::free(memory);
#endif
}

void *TensorArena::allocate(size_t bytes, size_t alignment) {
  uintptr_t current_ptr = reinterpret_cast<uintptr_t>(memory + offset);
  size_t padding = (alignment - (current_ptr % alignment)) % alignment;

  if (offset + padding + bytes > capacity) {
    return nullptr; // Out of memory
  }

  offset += padding;
  void *ptr = memory + offset;
  offset += bytes;
  return ptr;
}

void TensorArena::reset() { offset = 0; }

size_t TensorArena::get_usage() const { return offset; }
size_t TensorArena::get_capacity() const { return capacity; }
