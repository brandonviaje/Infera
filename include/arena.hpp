#ifndef ARENA_HPP
#define ARENA_HPP

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

class TensorArena {
public:
  /**
   * @param size_in_bytes Total pre-allocated pool size.
   */
  explicit TensorArena(size_t size_in_bytes);
  ~TensorArena();

  TensorArena(const TensorArena &) = delete;
  TensorArena &operator=(const TensorArena &) = delete;

  /**
   * Allocates a chunk of memory from arena.
   * @param bytes Number of bytes requested.
   * @param alignment Alignment boundary (default 64 for SIMD)
   */
  void *allocate(size_t bytes, size_t alignment = 64);

  /**
   * Resets bump pointer to zero. Doesn't wipe data
   */
  void reset();

  size_t get_usage() const;
  size_t get_capacity() const;

private:
  uint8_t *memory;
  size_t capacity;
  size_t offset;
};

class CudaArena {
private:
  float *d_base_ptr; 
  size_t total_size; 
  size_t offset;    

public:
  /**
   * Create GPU memory arena.
   *
   * @param size_bytes Total VRAM pool size to allocate
   */
  CudaArena(size_t size_bytes);

  ~CudaArena();

  CudaArena(const CudaArena &) = delete;
  CudaArena &operator=(const CudaArena &) = delete;

  /**
   * Allocates a chunk of GPU memory from the arena.
   *
   * @param bytes Number of bytes to allocate in device memory.
   * @return Device pointer offset from the base allocation.
   *
   */
  float *allocate(size_t bytes);

  /**
   * Resets GPU arena
   * Does not free memory; only resets internal offset.
   */
  void reset();
};
#endif
