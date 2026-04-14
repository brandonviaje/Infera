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

#endif
