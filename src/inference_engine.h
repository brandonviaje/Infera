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
    void compile(Graph& graph, const std::vector<std::vector<size_t>>& input_shapes);
    std::vector<Tensor<float>*> run(const std::vector<Tensor<float>*>& inputs);
private:
    struct OpNode 
    {
        Operator* op;
        std::vector<Tensor<float>*> inputs;
        std::vector<Tensor<float>*> outputs;
        std::string name;
    };
    std::vector<OpNode> execution_plan_;
    std::vector<std::unique_ptr<Operator>> op_store_;               // own operators so they persist between runs
    std::unordered_map<std::string, Tensor<float>*> symbol_table_;  // map "tensor_name" -> ptr to Tensor data
    std::vector<std::unique_ptr<Tensor<float>>> tensor_arena_;      // own the intermediate tensors created during inference.
    std::vector<std::string> graph_input_names_;                    // store input names
    std::vector<std::string> graph_output_names_;                   // store output names
};

#endif
