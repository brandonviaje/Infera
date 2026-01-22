#ifndef OPS_GEMM_H
#define OPS_GEMM_H

#include "../operator.h"
#include "../attribute.h"
#include "../tensor.h"
#include <cmath>

class GemmOperator : public Operator
{
public:
    // parse and set attributes
    void set_attributes(const Node& node) override 
    {
        // ONNX defaults: alpha=1.0, beta=1.0, transA=0, transB=0
        alpha_  = node.get_attribute<float>("alpha").value_or(1.0f);
        beta_   = node.get_attribute<float>("beta").value_or(1.0f);
        transA_ = node.get_attribute<int64_t>("transA").value_or(0);
        transB_ = node.get_attribute<int64_t>("transB").value_or(0);
    }

    // execute GEMM operation
    void forward(const std::vector<Tensor<float>*>& inputs, std::vector<Tensor<float>*>& outputs) override
    {
        // inputs
        const Tensor<float>* A = inputs[0];
        const Tensor<float>* B = inputs[1]; 
        const Tensor<float>* C = (inputs.size() > 2) ? inputs[2] : nullptr; // bias (optional)

        // determine dimensions based on transpose flags
        std::size_t M = transA_ ? A->shape()[1] : A->shape()[0]; 
        std::size_t K = transA_ ? A->shape()[0] : A->shape()[1]; 
        std::size_t N = transB_ ? B->shape()[0] : B->shape()[1]; 

        // prepare output
        Tensor<float>* Y = new Tensor<float>({M, N});
        outputs.push_back(Y);

        // init raw pointers 
        const float* a_ptr = A->data(); 
        const float* b_ptr = B->data(); 
        const float* c_ptr = C ? C->data() : nullptr;
        float* y_ptr       = Y->data();

        std::size_t a_width = A->shape()[1];
        std::size_t b_width = B->shape()[1];
        
        // GEMM loop
        for (std::size_t m = 0; m < M; ++m) 
        {
            for (std::size_t n = 0; n < N; ++n) 
            {
                float sum = 0.0f;

                for (std::size_t k = 0; k < K; ++k) 
                {
                    // handle transposes via index mapping
                    float val_a = transA_ ? a_ptr[k * a_width + m] : a_ptr[m * a_width + k]; // A[m, k] normally. If transA, we read A[k, m]
                    float val_b = transB_ ? b_ptr[n * b_width + k] : b_ptr[k * b_width + n]; // B[k, n] normally. If transB, we read B[n, k]

                    sum += val_a * val_b;
                }

                // add scaling with alpha
                sum *= alpha_;

                // apply bias if available
                if (c_ptr) 
                {
                    sum += beta_ * c_ptr[n];
                }
                y_ptr[m * N + n] = sum;
            }
        }
    }

private:
    float alpha_ = 1.0f;
    float beta_  = 1.0f;
    bool transA_ = false;
    bool transB_ = false;
};

#endif
