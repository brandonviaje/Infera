#ifndef OPS_ADD_H
#define OPS_ADD_H

#include "../operator.h"
#include <stdexcept>

class AddOperator : public Operator
{
public:
    void forward(const std::vector<Tensor<float>*>& inputs, std::vector<Tensor<float>*>& outputs) override
    {
        if (inputs.size() != 2) 
        {
            throw std::runtime_error("Add operator expects exactly 2 inputs.");
        }
        
        const Tensor<float>* A = inputs[0];
        const Tensor<float>* B = inputs[1];

        // check shape (ONNX does broadcasting, will impl later)
        if (A->shape() != B->shape()) 
        {
            throw std::runtime_error("Add operator currently requires inputs to have identical shapes.");
        }

        // prep output
        Tensor<float>* Y = new Tensor<float>(A->shape());
        outputs.push_back(Y);

        // raw pointers
        const float* a_ptr = A->data();
        const float* b_ptr = B->data();
        float* y_ptr       = Y->data();

        std::size_t total_elements = 1;

        for (std::size_t dim : A->shape()) 
        {
            total_elements *= dim;
        }

        for (std::size_t i = 0; i < total_elements; ++i) 
        {
            y_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    }
};

#endif
