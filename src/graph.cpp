#include "graph.h"
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

// constructor 
Graph::Graph(const onnx::GraphProto& graph_proto)
{
    // store input names
    inputs_.reserve(graph_proto.input_size());
    for (const auto& in : graph_proto.input())
        inputs_.push_back(in.name());

    // store output names
    outputs_.reserve(graph_proto.output_size());
    for (const auto& out : graph_proto.output())
        outputs_.push_back(out.name());

    // load weights into nodes
    for (const auto& tensor_proto : graph_proto.initializer())
    {
        std::vector<size_t> shape;
        for (auto dim : tensor_proto.dims()) shape.push_back(dim);

        auto tensor = std::make_unique<Tensor<float>>(shape);

        // load data ( onnx uses raw bytes or float list)
        if (tensor_proto.has_raw_data()) 
        {
            const std::string& raw = tensor_proto.raw_data();
            std::memcpy(tensor->data(), raw.data(), raw.size());
        } 
        else 
        {
            for (int i = 0; i < tensor_proto.float_data_size(); ++i)
                tensor->data()[i] = tensor_proto.float_data(i);
        }
        initializers_[tensor_proto.name()] = std::move(tensor);
    }

    // map tensor name -> node that produces it
    std::unordered_map<std::string, Node*> tensor_to_producer;

    // create nodes and register outputs
    node_map_.reserve(graph_proto.node_size());
    for (const auto& node_proto : graph_proto.node())
    {
        auto node = std::make_unique<Node>(node_proto);
        Node* node_ptr = node.get();
        std::string name = node->get_name();

        // handle nodes without a name
        if (name.empty())
            name = "node_" + std::to_string(node_map_.size());

        // register each output tensor
        for (const auto& output : node->get_outputs())
            tensor_to_producer[output] = node_ptr;

        // store the node in the map
        NodeInfo info;
        info.node = std::move(node);
        node_map_[name] = std::move(info);
    }

    // connect edges based on input/output tensors
    for (auto& [name, info] : node_map_)
    {
        Node* current_node = info.node.get();

        for (const auto& input_name : current_node->get_inputs())
        {
            if (tensor_to_producer.count(input_name))
            {
                Node* parent = tensor_to_producer[input_name];

                // link parent -> child
                node_map_[parent->get_name()].children.push_back(current_node);

                // link child -> parent
                info.parents.push_back(parent);
            }
        }
    }
}

// add new node to the graph
void Graph::add_node(std::unique_ptr<Node> node)
{
    std::string name = node->get_name();
    Node* ptr = node.get();

    NodeInfo info;
    info.node = std::move(node);
    node_map_[name] = std::move(info);

    // update edges after adding
    update_edges(ptr);
}

// update all edges of a node
void Graph::update_edges(Node* node)
{
    add_incoming_edges(node);
    add_outgoing_edges(node);
}

// add edges from nodes producing this node's inputs
void Graph::add_incoming_edges(Node* node)
{
    for (const auto& input : node->get_inputs())
    {
        for (auto& [name, info] : node_map_)
        {
            const auto& outputs = info.node->get_outputs();
            if (std::find(outputs.begin(), outputs.end(), input) != outputs.end())
            {
                info.children.push_back(node);
                node_map_[node->get_name()].parents.push_back(info.node.get());
            }
        }
    }
}

// add edges to nodes using this node's outputs
void Graph::add_outgoing_edges(Node* node)
{
    for (const auto& output : node->get_outputs())
    {
        for (auto& [name, info] : node_map_)
        {
            const auto& inputs = info.node->get_inputs();
            if (std::find(inputs.begin(), inputs.end(), output) != inputs.end())
            {
                node_map_[node->get_name()].children.push_back(info.node.get());
                info.parents.push_back(node);
            }
        }
    }
}

// replace node with a new one
void Graph::replace_node(Node* old_node, std::unique_ptr<Node> new_node)
{
    std::string old_name {old_node->get_name()};

    // check if node already exists
    if (node_map_.find(old_name) == node_map_.end())
        return;

    auto& info {node_map_[old_name]};
    new_node->set_name(old_name); 
    info.node = std::move(new_node);
}

// get input name by index
const std::string& Graph::get_input_name(std::size_t index) const
{
    return inputs_.at(index);
}

// get output name by index
const std::string& Graph::get_output_name(std::size_t index) const
{
    return outputs_.at(index);
}

// returns nodes in topological order
std::vector<Node*> Graph::topological_sort() 
{
    //  return if already sorted 
    if (!sorted_nodes_.empty()) 
    {
        return sorted_nodes_;
    }

    // track visited nodes + stack for dfs ordering
    std::unordered_set<Node*> visited;
    std::stack<Node*> stack;

    // start dfs from graph roots / input-connected nodes
    for (const auto& [_, info] : node_map_) 
    {
        if (info.parents.empty() || is_input_node(info.node.get())) 
        {
            topological_sort_util(info.node.get(), visited, stack);
        }
    }
    
    sorted_nodes_.reserve(stack.size());                    // reserve upfront to avoid reallocs

    // unwind stack into final sorted order
    while (!stack.empty()) 
    {
        sorted_nodes_.push_back(stack.top());
        stack.pop();
    }

    return sorted_nodes_;
}

// check if node consumes any graph-level input
bool Graph::is_input_node(Node* node) const
{
    // if any node input matches a graph input return true
    return std::any_of(
        node->get_inputs().begin(),
        node->get_inputs().end(),
        [this](const std::string& input) 
        {
            return std::find(inputs_.begin(), inputs_.end(), input) != inputs_.end();
        }
    );
}

// dfs helper for topological sort
void Graph::topological_sort_util(Node* node, std::unordered_set<Node*>& visited, std::stack<Node*>& stack)
{
    visited.insert(node);                       // mark node as visited

    // visit children first
    for (Node* child : node_map_[node->get_name()].children) 
    {
        if (visited.find(child) == visited.end()) 
        {
            topological_sort_util(child, visited, stack);
        }
    }
    
    stack.push(node); // push after children so dependencies come first
}

// print graph 
void Graph::print_graph() const
{
    std::cout << "graph topology:\n";
    for (const auto& [name, info] : node_map_)
    {
        std::cout << "  [" << name << "]";
        if (!info.children.empty())
        {
            std::cout << " -> ";
            // print node's childs
            for (auto* child : info.children)
                std::cout << child->get_name() << " ";
        }
        std::cout << "\n";
    }
}
