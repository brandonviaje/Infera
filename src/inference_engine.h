#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "graph.h"
#include "tensor.h"
#include "operator_registry.h" 

class InferenceEngine
{
public:
    InferenceEngine() = default;
    std::vector<Tensor<float>*> run(Graph& graph, const std::vector<Tensor<float>*>& inputs);
private:
    std::unordered_map<std::string, Tensor<float>*> symbol_table_;  // map "tensor_name" -> ptr to Tensor data
    std::vector<std::unique_ptr<Tensor<float>>> tensor_arena_;      // own the intermediate tensors created during inference.
};

#endif
