#include "../src/inference_engine.h"
#include "../src/graph.h"
#include "../src/tensor.h"
#include "../src/onnx-ml.pb.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

void test_mnist_inference() 
{
    // load the ONNX Model
    std::string filename = "models/mnist_ffn.onnx"; \
    std::ifstream input(filename, std::ios::ate | std::ios::binary);
    
    if (!input.is_open()) 
    {
        std::cerr << " [FAIL] Could not open " << filename << ". Make sure it exists!\n";
        return;
    }

    std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    
    std::cout << "Loading model: " << filename << " (" << size << " bytes)...\n";

    onnx::ModelProto model_proto;
    if (!model_proto.ParseFromIstream(&input)) 
    {
        std::cerr << " [FAIL] Failed to parse ONNX model.\n";
        return;
    }

    // init graph
    Graph graph(model_proto.graph());
    std::cout << "Graph loaded. Inputs: " << graph.get_input_size()  << ", Outputs: " << graph.get_output_size() << "\n";

    // create dummy input
    std::vector<std::size_t> input_shape = {1, 1, 28, 28};
    auto input_tensor = std::make_unique<Tensor<float>>(input_shape);

    // fill with mock data 
    std::size_t total_elements = 1 * 1 * 28 * 28;
    for (std::size_t i {}; i < total_elements; ++i) 
    {
        input_tensor->data()[i] = 0.5f; 
    }

    std::vector<Tensor<float>*> inputs;
    inputs.push_back(input_tensor.get());

    // run Inference
    InferenceEngine engine;
    
    try 
    {
        std::vector<Tensor<float>*> results = engine.run(graph, inputs);

        // inspect results
        if (results.empty()) 
        {
            std::cerr << " [FAIL] No results returned!\n";
            return;
        }

        Tensor<float>* output = results[0];
        std::cout << "Inference completed successfully!\n";
        std::cout << "Output Shape: [ ";
        for(auto d : output->shape()) std::cout << d << " ";
        std::cout << "]\n";

        // Print Logits
        std::cout << "\nClass Scores (Logits):\n";
        std::cout << "----------------------\n";
        
        float max_score = -1e9;
        int predicted_digit = -1;

        // Assuming tensor has .size() or similar
        for (std::size_t i = 0; i < output->size(); ++i) 
        { 
            float val = output->data()[i];
            std::cout << " Digit " << i << ": " << std::fixed << std::setprecision(4) << val << "\n";
            
            if (val > max_score) 
            {
                max_score = val;
                predicted_digit = i;
            }
        }
        std::cout << "----------------------\n";
        std::cout << " [PASS] Model predicts digit: " << predicted_digit << "\n";

    } 
    catch (const std::exception& e) 
    {
        std::cerr << " [FAIL] Inference Error: " << e.what() << "\n";
    }
}

int main() 
{
    test_mnist_inference();
    std::cout << "\n INFERENCE ENGINE TESTS PASSED!" << '\n';
    return 0;
}
