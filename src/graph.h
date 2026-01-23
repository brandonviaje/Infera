#ifndef GRAPH_H
#define GRAPH_H

#include <stack>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "tensor.h"
#include "node.h"
#include "onnx-ml.pb.h"

class Graph
{
public:
    struct NodeInfo
    {
        std::unique_ptr<Node> node;
        std::vector<Node*> children;
        std::vector<Node*> parents;
    };

    Graph() = default;
    Graph(const onnx::GraphProto& graph_proto);

    const std::string& get_input_name(std::size_t index) const;
    const std::string& get_output_name(std::size_t index) const;
    void print_graph() const;
    void add_node(std::unique_ptr<Node> node);
    void replace_node(Node* old_node, std::unique_ptr<Node> new_node);
    std::vector<Node*> topological_sort();

private:
    void update_edges(Node* node);
    void add_incoming_edges(Node* node);
    void add_outgoing_edges(Node* node);
    void topological_sort_util(Node* node, std::unordered_set<Node*>& visited, std::stack<Node*>& stack);
    bool is_input_node(Node* node) const;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, NodeInfo> node_map_;
    std::vector<Node*> sorted_nodes_;
    std::unordered_map<std::string, std::unique_ptr<Tensor<float>>> initializers_;
};

#endif
