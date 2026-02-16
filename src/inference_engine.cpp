#include "inference_engine.h"
#include "operator_registry.h"
#include <iostream>
#include <stdexcept>

// build execution plan + allocate tensors
void InferenceEngine::compile(Graph& graph, const std::vector<std::vector<size_t>>& input_shapes)
{
    std::cout << "[Compiler] Starting compilation...\n";

    // reset engine state
    execution_plan_.clear();
    op_store_.clear();
    tensor_arena_.clear();
    symbol_table_.clear();
    graph_input_names_.clear();
    graph_output_names_.clear();

    // validate input shape count
    if (input_shapes.size() != graph.get_input_size()) 
    {
        throw std::runtime_error("Input shape count mismatch");
    }

    // store graph input names + allocate input tensors
    for (std::size_t i = 0; i < graph.get_input_size(); ++i) 
    {
        std::string name = graph.get_input_name(i);
        graph_input_names_.push_back(name);

        auto tensor = std::make_unique<Tensor<float>>(input_shapes[i]);
        symbol_table_[name] = tensor.get();
        tensor_arena_.push_back(std::move(tensor));
    }

    // store graph output names
    for (std::size_t i = 0; i < graph.get_output_size(); ++i) 
    {
        graph_output_names_.push_back(graph.get_output_name(i));
    }

    auto sorted_nodes = graph.topological_sort(); // sort nodes

    // preload initializers (weights)
    for (const auto& node : sorted_nodes) 
    {
        for (const auto& input_name : node->get_inputs()) 
        {
            if (graph.has_initializer(input_name)) 
            {
                symbol_table_[input_name] = graph.get_initializer(input_name);
            }
        }
    }

    // build execution plan
    for (const auto* node : sorted_nodes)
    {
        std::string op_type = node->get_optype();

        auto op = OperatorRegistry::create_operator(op_type);
        if (!op) {
            std::cerr << "[Warning] Unknown operator: "
                      << op_type << "\n";
            continue;
        }

        op->set_attributes(*node);

        // resolve inputs
        std::vector<Tensor<float>*> op_inputs;
        for (const auto& name : node->get_inputs()) 
        {
            op_inputs.push_back(symbol_table_.at(name));
        }

        // alloc outputs
        std::vector<Tensor<float>*> op_outputs;
        for (const auto& name : node->get_outputs()) 
        {
            auto new_tensor = std::make_unique<Tensor<float>>();
            Tensor<float>* ptr = new_tensor.get();

            tensor_arena_.push_back(std::move(new_tensor));
            symbol_table_[name] = ptr;
            op_outputs.push_back(ptr);
        }

        // run shape inference 
        op->compute_output_shapes(op_inputs, op_outputs);

        // store execution step
        execution_plan_.push_back({
            op.get(),
            op_inputs,
            op_outputs,
            node->get_name()
        });

        op_store_.push_back(std::move(op));
    }

    std::cout << "[Compiler] Compilation complete. Nodes compiled: " << execution_plan_.size() << "\n";
}

std::vector<Tensor<float>*> InferenceEngine::run(const std::vector<Tensor<float>*>& inputs)
{
    // check if engine compiled first
    if (execution_plan_.empty()) 
    {
        throw std::runtime_error("Engine not compiled.");
    }

    if (inputs.size() != graph_input_names_.size()) throw std::runtime_error("Input size mismatch.");
    
    // copy runtime inputs into preallocated tensors
    for (std::size_t i = 0; i < inputs.size(); ++i) 
    {
        const std::string& name = graph_input_names_[i];
        *symbol_table_.at(name) = *inputs[i];
    }

    std::cout << "[Runtime] Executing compiled graph...\n";

    // exec plan 
    for (auto& step : execution_plan_) 
    {
        step.op->forward(step.inputs, step.outputs);
    }

    // build outputs
    std::vector<Tensor<float>*> results;
    for (const auto& name : graph_output_names_) 
    {
        results.push_back(symbol_table_.at(name));
    }

    return results;
}
