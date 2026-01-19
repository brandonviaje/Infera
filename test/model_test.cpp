#include <iostream>
#include <fstream>
#include <cassert>
#include <fstream>
#include "../src/onnx-ml.pb.h"
#include "../src/loader.h"
#include "../src/tensor.h"

void test_protobuf_instantiation()
{
    onnx::ModelProto model;
    model.set_ir_version(8);
    assert(model.ir_version() == 8);
    std::cout << "ONNX ModelProto instantiated and checked!" << std::endl;
}

void test_load_onnx_tensor()
{
    // ceate dummy ONNX TensorProto
    onnx::TensorProto proto;
    proto.add_dims(2);
    proto.add_dims(2);
    proto.set_data_type(onnx::TensorProto::FLOAT);

    std::vector<float> original_data = {1.1f, 2.2f, 3.3f, 4.4f};

    // copy floats into raw byte string
    proto.set_raw_data(std::string(reinterpret_cast<const char *>(original_data.data()), original_data.size() * sizeof(float)));

    // use Loader
    Tensor<float> tensor = Loader::load_tensor(proto);

    // assertions
    assert(tensor.shape()[0] == 2);
    assert(tensor.shape()[1] == 2);
    assert(tensor.at({0, 0}) == 1.1f);
    assert(tensor.at({1, 1}) == 4.4f);

    std::cout << "ONNX Tensor conversion passed!" << std::endl;
}

void test_file_loading()
{
    std::string path{"models/mnist.onnx"};
    std::ifstream input(path, std::ios::binary);

    // check if can't find file
    if (!input.is_open())
    {
        std::cout << "[INFO] Skipping random file test (file '" << path << "' not found)." << std::endl;
        return;
    }

    // parse  File
    onnx::ModelProto model;

    if (!model.ParseFromIstream(&input))
    {
        throw std::runtime_error("Failed to parse ONNX file");
    }

    const onnx::GraphProto &graph{model.graph()};

    // load every supported tensor
    int success_count{};

    for (int i = 0; i < graph.initializer_size(); ++i)
    {
        const onnx::TensorProto &tensor_proto = graph.initializer(i);

        // only load float types for now
        if (tensor_proto.data_type() == onnx::TensorProto::FLOAT)
        {
            try
            {
                Tensor<float> t = Loader::load_tensor(tensor_proto);

                // assertions
                assert(t.size() > 0);
                assert(!t.shape().empty());
                success_count++;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to load tensor " << tensor_proto.name() << ": " << e.what() << std::endl;
                throw;
            }
        }
    }
    std::cout << "Successfully loaded " << success_count << " FLOAT tensors." << std::endl;
}

int main()
{
    try
    {
        test_protobuf_instantiation();
        test_load_onnx_tensor();
        test_file_loading();
        std::cout << "MODEL TESTS PASSED!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Model test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}