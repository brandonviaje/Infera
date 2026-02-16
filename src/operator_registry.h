#ifndef REGISTRY_H
#define REGISTRY_H

#include <string>
#include <memory>
#include <iostream>

#include "operator.h"
#include "ops/flatten.h"
#include "ops/gemm.h"
#include "ops/relu.h"
#include "ops/add.h"
#include "ops/conv.h"
#include "ops/maxpool.h"
#include "ops/reshape.h"

class OperatorRegistry
{
public:
    // simple factory design pattern
    static std::unique_ptr<Operator> create_operator(const std::string& type)
    {
        if (type == "Add")          return std::make_unique<AddOperator>();
        if (type == "Flatten")      return std::make_unique<FlattenOperator>();
        if (type == "MaxPool")      return std::make_unique<MaxPoolOperator>();
        if (type == "Conv")         return std::make_unique<ConvOperator>();
        if (type == "Reshape")      return std::make_unique<ReshapeOperator>();
        if (type == "Gemm")         return std::make_unique<GemmOperator>();
        if (type == "Relu")         return std::make_unique<ReluOperator>();
        std::cerr << "Warning: Operator '" << type << "' not implemented yet." << std::endl;
        return nullptr;
    }
};

#endif
