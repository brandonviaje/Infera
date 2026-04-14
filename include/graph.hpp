#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "tensor.hpp"
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace infera {

// Node interface
class Node {
public:
  std::string name;
  std::vector<std::string> input_tensors;
  std::vector<std::string> output_tensors;

  Node(const std::string &name, const std::vector<std::string> &inputs,
       const std::vector<std::string> &outputs)
      : name(name), input_tensors(inputs), output_tensors(outputs) {}

  virtual ~Node() = default;
  virtual void forward() = 0;
};

// Operation Nodes - wrappers around CUDA kernels.

class MatMulNode : public Node {
private:
  float *d_A; // device ptr to input A
  float *d_B; // device ptr to weights B
  float *d_C; // device ptr to output C
  int M, K, N;

public:
  MatMulNode(std::string name, std::vector<std::string> inputs,
             std::vector<std::string> outputs, float *d_A, float *d_B,
             float *d_C, int M, int K, int N)
      : Node(name, inputs, outputs), d_A(d_A), d_B(d_B), d_C(d_C), M(M), K(K),
        N(N) {}

  void forward() override;
};

class AddNode : public Node {
private:
  float *d_A;
  float *d_B;
  float *d_C;
  int num_elements;

public:
  AddNode(std::string name, std::vector<std::string> inputs,
          std::vector<std::string> outputs, float *d_A, float *d_B, float *d_C,
          int num_elements)
      : Node(name, inputs, outputs), d_A(d_A), d_B(d_B), d_C(d_C),
        num_elements(num_elements) {}

  void forward() override;
};

class ReLUNode : public Node {
private:
  float *d_X;
  float *d_Y;
  int num_elements;

public:
  ReLUNode(std::string name, std::vector<std::string> inputs,
           std::vector<std::string> outputs, float *d_X, float *d_Y,
           int num_elements)
      : Node(name, inputs, outputs), d_X(d_X), d_Y(d_Y),
        num_elements(num_elements) {}

  void forward() override;
};

// Computational Graph - owns the exec sequence and manages GPU memory
class Graph {
private:
  std::unordered_map<std::string, std::unique_ptr<Node>> nodes;
  std::unordered_map<std::string, std::vector<std::string>> adj_list;
  std::unordered_map<std::string, int> in_degrees;

public:
  Graph() = default;

  void add_node(std::unique_ptr<Node> node);
  void build_dag();
  void execute();
};

} // namespace infera

#endif
