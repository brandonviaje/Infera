#ifndef OPERATOR_H
#define OPERATOR_H

#include <vector>
#include <string>
#include "tensor.h"
#include "node.h"

class Operator {
public:
    // virtual destructor
    virtual ~Operator() = default;

    // execute operator
    virtual void compute(const Node& node, const std::vector<Tensor<float>*>& inputs, Tensor<float>& output) = 0;
};

#endif
