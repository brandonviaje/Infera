#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath> 
#include "graph.h"
#include "onnx_parser.h"
#include "image_loader.h"
#include "inference_engine.h"
#include "tensor.h"

int main(int argc, char** argv)
{
    if (argc < 3) 
    {
        std::cerr << "Usage: ./infera <model.onnx> <image.png>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    try {
        // build Graph
        Graph graph;
        OnnxParser parser;

        std::cout << "Loading Model: " << model_path << "...\n";
        parser.parse(graph, model_path);

        // detect size dynamically
        graph.infer_input_size();

        int req_h = graph.get_input_height();
        int req_w = graph.get_input_width();

        std::cout << "Model requires: " << req_w << "x" << req_h << "\n";

        // load image
        std::cout << "Loading Image: " << image_path << "...\n";
        Tensor<float>* input_tensor = ImageLoader::load_image(image_path, req_w, req_h);

        InferenceEngine engine;
        std::cout << "Running Inference...\n";

        std::vector<Tensor<float>*> inputs = { input_tensor };
        std::vector<Tensor<float>*> outputs = engine.run(graph, inputs);

        if (outputs.empty())
            throw std::runtime_error("No output from engine.");

        Tensor<float>* result = outputs[0];
        const float* output_data = result->data();

        // results using softmax
        std::cout << "\n=== Results ===\n";
        
        float sum {};
        std::vector<float> exp_values(10);
        
        // get max logit for numerical stability
        float max_logit = -1e9f;
        for (int i {}; i < 10; ++i) 
        {
            if (output_data[i] > max_logit) max_logit = output_data[i];
        }

        // exponentiate and sum
        for (int i = 0; i < 10; ++i) 
        {
            exp_values[i] = std::exp(output_data[i] - max_logit);
            sum += exp_values[i];
        }

        int predicted_digit = -1;
        float max_prob = -1.0f;

        for (int i {}; i < 10; ++i) 
        {
            float probability = exp_values[i] / sum;
            float percent = probability * 100.0f;
            std::cout << "Digit " << i << ": " << percent << "%\n";

            if (probability > max_prob) 
            {
                max_prob = probability;
                predicted_digit = i;
            }
        }

        std::cout << "\nPREDICTION: " << predicted_digit  << " (Confidence: " << (int)(max_prob * 100) << "%)\n";
        delete input_tensor; 
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
