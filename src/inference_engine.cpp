#include "inference_engine.h"
#include "operator_registry.h" 
#include <iostream>
#include <stdexcept>

std::vector<Tensor<float>*> InferenceEngine::run(Graph& graph, const std::vector<Tensor<float>*>& inputs) 
{
    // reset state
    symbol_table_.clear();
    tensor_arena_.clear();

    // make sure input counts match 
    if (inputs.size() != graph.get_input_size()) 
    {
        throw std::runtime_error("input size mismatch! graph expects " + std::to_string(graph.get_input_size()) + " inputs but got " + std::to_string(inputs.size()));
    }

    // map graph input names to input tensors
    for (std::size_t i {}; i < inputs.size(); ++i) 
    {
        const std::string& name = graph.get_input_name(i);
        symbol_table_[name] = inputs[i];
    }

    // grab nodes in topological order
    auto sorted_nodes = graph.topological_sort();

    // preload all initializer tensors 
    for (const auto& node : sorted_nodes) 
    {
        for (const auto& input_name : node->get_inputs()) 
        {
            // if the graph owns this tensor, pull it in
            if (graph.has_initializer(input_name)) 
            {
                symbol_table_[input_name] = graph.get_initializer(input_name);
            }
        }
    }

    std::cout << "starting inference engine...\n";

    // execution loop
    for (Node* node : sorted_nodes) 
    {
        std::string op_type = node->get_optype(); 
        std::cout << "Running Node: " << node->get_name() << " [" << op_type << "]" << std::endl;

        // create the operator from the registry
        auto op = OperatorRegistry::create_operator(op_type);

        if (!op)  
        {
            std::cerr << " [warning] no implementation for operator: " << op_type << " (skipping)\n";
            continue;
        }

        op->set_attributes(*node);

        // collect input tensors for this operator

        std::vector<Tensor<float>*> op_inputs;
        for (const auto& name : node->get_inputs()) 
        {
            if (symbol_table_.find(name) == symbol_table_.end()) 
            {
                throw std::runtime_error("runtime error: missing dependency '" + name + "' for node " + node->get_name());
            }
            op_inputs.push_back(symbol_table_[name]);
        }

        // allocate output tensors and register them
        std::vector<Tensor<float>*> op_outputs;

        for (const auto& name : node->get_outputs()) 
        {
            auto new_tensor = std::make_unique<Tensor<float>>(std::vector<std::size_t>{});
            Tensor<float>* ptr = new_tensor.get();

            tensor_arena_.push_back(std::move(new_tensor));
            symbol_table_[name] = ptr;
            op_outputs.push_back(ptr);
        }

        op->forward(op_inputs, op_outputs);
    }

    // collect final graph outputs 
    std::vector<Tensor<float>*> final_results;

    // loop by index since output names are index-based
    for (std::size_t i {}; i < graph.get_output_size(); ++i) 
    {
        const std::string& name = graph.get_output_name(i);

        if (symbol_table_.count(name)) 
        {
            final_results.push_back(symbol_table_[name]);
        } 
        else 
        {
            throw std::runtime_error("graph output '" + name + "' was not produced during inference.");
        }
    }

    return final_results;
}
