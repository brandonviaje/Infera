#ifndef LOADER_H
#define LOADER_H

#include "onnx-ml.pb.h"
#include "tensor.h"
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

class Loader
{
public:
    static Tensor<float> load_tensor(const onnx::TensorProto &proto)
    {
        // get tensor shape from ONNX metadata
        std::vector<std::size_t> shape;

        for (int i = 0; i < proto.dims_size(); ++i)
        {
            shape.push_back(static_cast<size_t>(proto.dims(i)));
        }

        Tensor<float> tensor(shape);

        // currently only supporting FLOAT tensors, will handle diff types later
        if (proto.data_type() != onnx::TensorProto::FLOAT)
        {
            throw std::runtime_error("Only FLOAT tensors are currently supported");
        }

        // ONNX tensors store data either as raw bytes or as typed fields
        if (proto.has_raw_data())
        {
            const std::string &raw{proto.raw_data()};

            // check byte size before copying
            if (raw.size() != tensor.size() * sizeof(float))
            {
                throw std::runtime_error("Raw data size mismatch");
            }

            // copy from raw buffer
            std::memcpy(tensor.data(), raw.data(), raw.size());
        }
        else if (proto.float_data_size() > 0)
        {
            // for non-packed float storage
            for (int i = 0; i < proto.float_data_size(); ++i)
            {
                tensor.data()[i] = proto.float_data(i);
            }
        }
        return tensor;
    }
};

#endif
