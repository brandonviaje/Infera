#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <numeric>
#include <string>
#include <stdexcept>
#include <vector>

template <typename T>
class Tensor
{
public:
    // constructors
    Tensor() : data_(nullptr), size_(0) {}

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
    Tensor(const Tensor &other) : shape_(other.shape_), size_(other.size_)
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

    // copy-assignment operator
    Tensor &operator=(const Tensor &other)
    {

        // self-assignment check
        if (this != &other)
        {
            // clean existing memory
            delete[] data_;

            // copy metadata
            shape_ = other.shape_;
            size_ = other.size_;

            // alloc and copy new data
            data_ = new T[size_];
            std::copy(other.data_, other.data_ + other.size_, data_);
        }
        return *this;
    }

    // move-assignment operator
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

    // getters
    std::size_t size() const { return size_; }
    const std::vector<std::size_t> &shape() const { return shape_; }

    T *data() { return data_; }
    const T *data() const { return data_; }

    T &operator[](std::size_t index) { return data_[index]; }
    const T &operator[](std::size_t index) const { return data_[index]; }

    std::size_t rows() const
    {
        return shape_.empty() ? 0 : shape_[0];
    }

    std::size_t cols() const
    {
        return shape_.size() < 2 ? 1 : shape_[1];
    }

    void reshape(const std::vector<std::size_t>& new_shape) {
        // get total size of new shape
        std::size_t new_total_size{1};
        
        for (auto dim : new_shape) 
        {
            new_total_size *= dim;
        }

        // check if total elements match
        if (new_total_size != size_) 
        {
            throw std::invalid_argument("Reshape error: Total element count must not change.");
        }

        // update shape
        shape_ = new_shape;
    }


    // multi-dimension getter
    T &at(const std::vector<std::size_t> &indices)
    {
        if (indices.size() != shape_.size())
        {
            throw std::invalid_argument("Dimension mismatch");
        }

        std::size_t offset = 0;
        std::size_t stride = 1;

        // iterate backwards through dimensions to calculate row-major offset
        for (long long i = shape_.size() - 1; i >= 0; --i)
        {
            if (indices[i] >= shape_[i])
            {
                throw std::out_of_range("Index out of bounds");
            }
            offset += indices[i] * stride;
            stride *= shape_[i];
        }
        return data_[offset];
    }

    // read values from at func
    const T &at(const std::vector<std::size_t> &indices) const
    {
        return const_cast<Tensor *>(this)->at(indices);
    }

private:
    T *data_;
    std::vector<std::size_t> shape_;
    std::size_t size_;
};

#endif
