#include "graph.hpp"
#include "ops.hpp"
#include <iostream>
#include <queue>
#include <stdexcept>

namespace infera {

// kernel wrappers
void MatMulNode::forward() {
  ops::launch_matmul_kernel(d_A, d_B, d_C, M, K, N);
}

void AddNode::forward() { ops::launch_add_kernel(d_A, d_B, d_C, num_elements); }

void ReLUNode::forward() { ops::launch_relu_kernel(d_X, d_Y, num_elements); }

// graph management and dag compilation

void Graph::add_node(std::unique_ptr<Node> node) {
  std::string node_name = node->name;
  nodes[node_name] = std::move(node);
}

void Graph::build_dag() {
  std::unordered_map<std::string, std::string> tensor_to_producer;

  // map output tensors to the node that produces them
  for (const auto &pair : nodes) {
    for (const std::string &out_tensor : pair.second->output_tensors) {
      tensor_to_producer[out_tensor] = pair.second->name;
    }
  }

  // build adj list and calculate dependencies
  for (const auto &pair : nodes) {
    auto &node = pair.second;
    int actual_dependencies = 0;

    for (const std::string &in_tensor : node->input_tensors) {
      if (tensor_to_producer.count(in_tensor)) {
        std::string producer_name = tensor_to_producer[in_tensor];
        adj_list[producer_name].push_back(node->name);

        actual_dependencies++;
      }
    }

    // Lock in the accurate dependency count
    in_degrees[node->name] = actual_dependencies;
  }

  std::cout << "[Graph] DAG compiled. Dependency map built successfully.\n";
}

// execution

void Graph::execute() {
  std::unordered_map<std::string, int> current_in_degrees = in_degrees;
  std::queue<std::string> ready_queue;

  // find all starting nodes
  for (const auto &pair : current_in_degrees) {
    if (pair.second == 0) {
      ready_queue.push(pair.first);
    }
  }

  // process the queue using Kahn's Topological Sort
  int executed_nodes = 0;
  while (!ready_queue.empty()) {
    std::string current_name = ready_queue.front();
    ready_queue.pop();

    nodes[current_name]->forward();
    executed_nodes++;

    // notify downstream nodes that this tensor is ready
    for (const std::string &neighbor : adj_list[current_name]) {
      current_in_degrees[neighbor]--;

      // if a downstream node has all its dependencies met, queue it up
      if (current_in_degrees[neighbor] == 0) {
        ready_queue.push(neighbor);
      }
    }
  }

  // safety check: did every node run
  if (executed_nodes != nodes.size()) {
    throw std::runtime_error("FATAL: Graph execution stuck! Check ONNX for "
                             "cyclic dependencies or missing tensors.");
  }
}

} // namespace infera
