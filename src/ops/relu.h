#ifndef OPS_RELU_H
#define OPS_RELU_H

#include "../operator.h"

class ReluOperator : public Operator
{
public:
    void compute([[maybe_unused]]const Node& node, const std::vector<Tensor<float>*>& inputs, Tensor<float>& output) override
    {
        const Tensor<float>* input {inputs[0]};
        output = Tensor<float>(input->shape()); // set output shape to same shape as input

        const float* in_ptr {input->data()};
        float* out_ptr {output.data()};
        std::size_t total_elements {input->size()};

        // apply ReLU to elem-wise
        for (std::size_t i{}; i < total_elements; ++i) 
        {
            float val {in_ptr[i]};
            out_ptr[i] = (val > 0.0f) ? val : 0.0f;   // ReLU: f(x) = max(0,x)
        }
    }
};

#endif
