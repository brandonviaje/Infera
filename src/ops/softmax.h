#ifndef OPS_SOFTMAX_H
#define OPS_SOFTMAX_H

#include "../operator.h"
#include <cmath>      
#include <algorithm>  
#include <limits>     

class SoftmaxOperator : public Operator
{
public:
    void compute([[maybe_unused]] const Node& node, const std::vector<Tensor<float>*>& inputs, Tensor<float>& output) override
    {
        const Tensor<float>* input {inputs[0]};
        const std::vector<std::size_t>& shape {input->shape()};
         
        output = Tensor<float>{shape};                   // set output shape same as input shape

        std::size_t features {shape[shape.size() - 1]};  // axis defaults to last dim
        std::size_t batch {input->size() / features};    // batch = outer dims, features = last dim
        
        const float* in_ptr {input->data()};
        float* out_ptr {output.data()};

        // iterate over each batch item
        for (std::size_t b {}; b < batch; ++b)
        {
            const float* row_in {in_ptr + b * features};
            float* row_out {out_ptr + b * features};

            // get max for numerical stability
            float max_val {*std::max_element(row_in, row_in + features)};

            // exponential and sum
            float sum {};

            for (size_t i {0}; i < features; ++i)
            {
                float val {std::exp(row_in[i] - max_val)};
                row_out[i] = val;
                sum += val;
            }

            // normalize
            for (size_t i {0}; i < features; ++i)
            {
                row_out[i] /= sum;
            }
        }
    }
};

#endif
