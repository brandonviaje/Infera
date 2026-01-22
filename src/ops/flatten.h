#ifndef OPS_FLATTEN_H
#define OPS_FLATTEN_H

#include "../operator.h"

class FlattenOperator : public Operator 
{
public:
    void set_attributes(const Node& node) override 
    {
        // ONNX Default is axis=1 
        axis_ = node.get_attribute<int64_t>("axis").value_or(1);
    }

    void forward(const std::vector<Tensor<float>*>& inputs, std::vector<Tensor<float>*>& outputs) override 
    {
        const Tensor<float>* X = inputs[0];

        // calculate new shape

        // handle negative axes (-1 means last dim)
        int64_t rank = static_cast<int64_t>(X->shape().size());
        int64_t axis = axis_;
        if (axis < 0) axis += rank;

        // clamp just in case
        if (axis < 0) axis = 0;
        if (axis > rank) axis = rank;

        // dim 1: product of all dims BEFORE axis
        std::size_t dim_0 = 1;
        for (int64_t i = 0; i < axis; ++i) {
            dim_0 *= X->shape()[i];
        }

        // dim 2: product of all dims AFTER axis
        std::size_t dim_1 = X->size() / dim_0;     // shortcut: total size / dim_0

        // prepare output
        Tensor<float>* Y = new Tensor<float>(*X);  // copy data
        Y->reshape({dim_0, dim_1});                // reshape output tensor
        
        outputs.push_back(Y);
    }

private:
    int64_t axis_ = 1;
};

#endif
