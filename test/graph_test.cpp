#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <algorithm>
#include "../src/graph.h"
#include "../src/node.h"
#include "../src/onnx-ml.pb.h"

// build dummy onnx::NodeProto
onnx::NodeProto create_node_proto(const std::string& name, const std::string& op_type, const std::vector<std::string>& inputs,  const std::vector<std::string>& outputs) 
{
    onnx::NodeProto node;
    node.set_name(name);
    node.set_op_type(op_type);

    // extract inputs
    for (const auto& in : inputs) node.add_input(in);

    // extract outputs
    for (const auto& out : outputs) node.add_output(out);

    return node;
}

void test_graph_construction() 
{
    std::cout << "Running Graph Construction Test...\n";

    // mock diamond graph
    onnx::GraphProto graph_proto;
    graph_proto.set_name("TestDiamondGraph");

    // init graph input
    auto* input = graph_proto.add_input();
    input->set_name("Input_Data");

    // init graph output
    auto* output = graph_proto.add_output();
    output->set_name("Output_Data");

    // Node A: Entry point (Conv)
    *graph_proto.add_node() = create_node_proto("Node_A", "Conv", {"Input_Data"}, {"Val_A"});

    // Node B: Left branch (Relu)
    *graph_proto.add_node() = create_node_proto("Node_B", "Relu", {"Val_A"}, {"Val_B"});

    // Node C: Right branch (Relu)
    *graph_proto.add_node() = create_node_proto("Node_C", "Relu", {"Val_A"}, {"Val_C"});

    // Node D: Merge point (Add)
    *graph_proto.add_node() = create_node_proto("Node_D", "Add", {"Val_B", "Val_C"}, {"Output_Data"});

    // build graph
    Graph graph(graph_proto);

    // assertions 
    std::cout << "  Verifying Inputs... ";
    assert(graph.get_input_name(0) == "Input_Data");
    std::cout << "OK\n";

    std::cout << "  Verifying Outputs... ";
    assert(graph.get_output_name(0) == "Output_Data");
    std::cout << "OK\n";

    std::cout << "\n  [Visual Check Required] Verify structure below:\n";
    std::cout << "  - Node_A should point to Node_B and Node_C\n";
    std::cout << "  - Node_B & Node_C should point to Node_D\n";
    std::cout << "---------------------------------------------------\n";
    
    graph.print_graph();
    
    std::cout << "---------------------------------------------------\n";
}

void test_load_from_file() {
    std::cout << "\nRunning File Load Test...\n";
    
    std::string path = "models/mnist_ffn.onnx";
    std::ifstream input_file(path, std::ios::binary);

    if (!input_file.is_open()) 
    {
        std::cerr << "  [SKIP] Could not open " << path << ".\n";
        return;
    }

    // load model proto
    onnx::ModelProto model_proto;
    if (!model_proto.ParseFromIstream(&input_file)) 
    {
        throw std::runtime_error("Failed to parse ONNX file");
    }

    std::cout << "  File loaded successfully.\n";

    Graph graph(model_proto.graph());                                       // extract graph proto and build graph

    // assert
    std::cout << "  Graph Name: " << model_proto.graph().name() << "\n";
    std::cout << "  Inputs: " << graph.get_input_name(0) << "\n";
    graph.print_graph(); 
    std::cout << "  [PASS] File loading successful.\n";
}

int main() 
{
    try 
    {
        test_graph_construction();
        test_load_from_file();
        std::cout << "\nGRAPH TESTS PASSED!\n";
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Graph test failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
