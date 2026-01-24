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
#include "ops/add.h"

class OperatorRegistry
{
public:
    //  simple factory design pattern
    static std::unique_ptr<Operator> create_operator(const std::string& type)
    {
        if(type == "Add")
        {
            return std::make_unique<AddOperator>();
        }
        if (type == "Flatten") 
        {
            return std::make_unique<FlattenOperator>();
        }
        else if (type == "Gemm")
        {
            return std::make_unique<GemmOperator>();
        }
        else if (type == "Relu")
        {
            return std::make_unique<ReluOperator>();
        }
        std::cerr << "Warning: Operator '" << type << "' not implemented yet." << std::endl; // else operator isn't registered/supported yet
        return nullptr;
    }
};

#endif
