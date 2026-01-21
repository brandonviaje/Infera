#ifndef OPS_GEMM_H
#define OPS_GEMM_H

#include "../operator.h"

class GemmOperator : public Operator
{
public:
    void compute([[maybe_unused]]const Node& node, const std::vector<Tensor<float>*>& inputs, Tensor<float>& output) override
    {
        // inputs
        const Tensor<float>* A = inputs[0];
        const Tensor<float>* B = inputs[1]; 
        const Tensor<float>* C = (inputs.size() > 2) ? inputs[2] : nullptr; // optional

        // give attributes default values
        float alpha{node.get_attribute("alpha").f()};
        float beta{node.get_attribute("beta").f()};
        bool transA{node.get_attribute("transA").i() == 1};
        bool transB{node.get_attribute("transB").i() == 1};

        // treat missing alpha/beta as 1.0
        if (alpha == 0.0f) alpha = 1.0f;
        if (beta  == 0.0f) beta  = 1.0f;

        // calculate dimensions : if transA is true swap rows/cols 
        std::size_t M{transA ? A->shape()[1] : A->shape()[0]}; // output rows
        std::size_t K{transA ? A->shape()[0] : A->shape()[1]}; // inner dimensions
        std::size_t N{transB ? B->shape()[0] : B->shape()[1]}; // output cols

        // allocate output
        output = Tensor<float>({M, N});

        // pointers memory access
        const float* a_ptr = A->data(); 
        const float* b_ptr = B->data(); 
        const float* c_ptr = C ? C->data() : nullptr;
        float* y_ptr = output.data();

        // GEMM 
        for (size_t m = 0; m < M; ++m) 
        {
            for (size_t n = 0; n < N; ++n) 
            {

                float sum = 0.0f;

                for (size_t k = 0; k < K; ++k) 
                {
                    // logical transpose matrix
                    std::size_t a_width {A->shape()[1]};
                    float val_a {transA ? a_ptr[k * a_width + m] : a_ptr[m * a_width + k]};

                    std::size_t b_width {B->shape()[1]};
                    float val_b {transB ? b_ptr[n * b_width + k] : b_ptr[k * b_width + n]};

                    sum += val_a * val_b;
                }

                // scale matmul
                sum *= alpha;

                // add bias if present 
                if (c_ptr) 
                {
                    sum += beta * c_ptr[n];
                }

                y_ptr[m * N + n] = sum;
            }
        }
    }
};

#endif
