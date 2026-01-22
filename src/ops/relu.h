#ifndef OPS_RELU_H
#define OPS_RELU_H

#include "../operator.h"

class ReluOperator : public Operator
{
public:
    void forward(const std::vector<Tensor<float>*>& inputs, std::vector<Tensor<float>*>& outputs) override
    {
        const Tensor<float>* X = inputs[0];

        // prepare output
        Tensor<float>* Y = new Tensor<float>(X->shape());
        outputs.push_back(Y);

        // ptrs to data
        const float* x_ptr = X->data();
        float* y_ptr       = Y->data();
        std::size_t size   = X->size();

        // apply relu element-wise
        
        for (std::size_t i{}; i < size; ++i) 
        {
            float val = x_ptr[i];
            y_ptr[i] = (val > 0.0f) ? val : 0.0f; // f(x) = max(0, x)
        }
    }
};

#endif
