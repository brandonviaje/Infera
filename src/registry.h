#ifndef REGISTRY_H
#define REGISTRY_H

#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>

#include "operator.h"
#include "ops/flatten.h"
#include "ops/gemm.h"
#include "ops/relu.h"
#include "ops/softmax.h"
#include "ops/sigmoid.h"

class Registry
{
public:
    //  simple factory design pattern
    static std::unique_ptr<Operator> create(const std::string& type)
    {
        if (type == "Flatten") 
        {
            return std::make_unique<FlattenOperator>();
        }
        else if (type == "Gemm")
        {
            return std::make_unique<GemmOperator>();
        }
        else if(type == "Sigmoid")
        {
            return std::make_unique<SigmoidOperator>();
        }
        else if (type == "Relu")
        {
            return std::make_unique<ReluOperator>();
        }
        else if (type == "Softmax")
        {
            return std::make_unique<SoftmaxOperator>();
        }
         
        std::cerr << "Warning: Operator '" << type << "' not implemented yet." << std::endl; // else operator isn't registered/supported yet
        return nullptr;
    }
};

#endif
