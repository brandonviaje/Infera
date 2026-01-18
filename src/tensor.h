#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
class Tensor
{
public:
    // constructors
    Tensor(const std::vector<size_t> &shape) : shape_(shape)
    {
        size_ = 1;
        for (auto dim : shape_)
            size_ *= dim;
        data_ = new T[size_];
    }

    // destructors
    ~Tensor()
    {
        delete[] data_;
    }

    // copy constructor
    Tensor(const Tensor &other) : _shape(other.shape), size_(other.size_)
    {
        data_ = new T[size_];
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // move constructor
    Tensor(Tensor &&other) noexcept : data_(other.data_), shape_(std::move(other.shape_)), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // copy-assignment constructor
    Tensor &operator=(const Tensor &other)
    {

        // self-assignment check
        if (this != &other)
        {
            // clean existing  mem
            delete[] data;

            // copy metadata
            shape_ = other.shape_;
            size_ = other.size_;

            // alloc and copy new data
            data_ = new T[size_];
            std::copy(other.data_, other.data_ + other.size_, data_);
        }
        return *this;
    }

    Tensor &operator=(Tensor &&other) noexcept
    {
        // self-assignment check
        if (this != &other)
        {
            // clean up existing memory
            delete[] data_;

            // move resources
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            size_ = other.size_;

            // reset source object
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

private:
    T *data_;
    std::vector<std::size_t> shape_;
    std::size_t size_;
};

#endif