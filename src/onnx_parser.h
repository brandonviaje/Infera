#ifndef ONNX_PARSER_H
#define ONNX_PARSER_H

#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "onnx-ml.pb.h" 
#include "graph.h"

class OnnxParser 
{
public:
    void parse(Graph& graph, const std::string& model_path)
    {
        std::ifstream input(model_path, std::ios::ate | std::ios::binary);
        if (!input.is_open()) throw std::runtime_error("Failed to open: " + model_path);

        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!input.read(buffer.data(), size)) throw std::runtime_error("Failed to read file");

        onnx::ModelProto model_proto;
        if (!model_proto.ParseFromArray(buffer.data(), size)) 
        {
            throw std::runtime_error("Failed to parse ONNX protobuf");
        }

        const onnx::GraphProto& graph_proto = model_proto.graph();
        std::cout << "Parsing Graph: " << graph_proto.name() << "\n";

        // add input and output

        for (const auto& input : graph_proto.input()) 
        {
            graph.add_input(input.name()); 
        }

        for (const auto& output : graph_proto.output()) 
        {
            graph.add_output(output.name());
        }
    
        // load initializers
        for (const auto& initializer : graph_proto.initializer()) 
        {
            std::vector<size_t> dims;
            for (auto d : initializer.dims()) dims.push_back(d);
            

            auto* tensor = new Tensor<float>(dims);
            

            if (initializer.has_raw_data())  // handle raw data
            {
                const std::string& raw = initializer.raw_data();
                std::memcpy(tensor->data(), raw.data(), raw.size());
            }  
            else // handle float data
            {
                for(int i=0; i<initializer.float_data_size(); ++i) 
                {
                    tensor->data()[i] = initializer.float_data(i);
                }
            }
            
            graph.add_initializer(initializer.name(), tensor); 
        }

        // load nodes
        for (const auto& node_proto : graph_proto.node()) 
        {
            auto node = std::make_unique<Node>(node_proto);
            graph.add_node(std::move(node));
        }
    }
};

#endif
