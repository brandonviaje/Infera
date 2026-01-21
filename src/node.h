#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include "onnx-ml.pb.h"

class Node
{
public:
    explicit Node(const onnx::NodeProto &proto)
    {
        name_ = proto.name();
        op_type_ = proto.op_type();

        for (const auto &input : proto.input())
            inputs_.push_back(input);
        for (const auto &output : proto.output())
            outputs_.push_back(output);

        // parse attributes
        for (const auto &attr : proto.attribute())
        {
            attributes_[attr.name()] = attr;
        }
    }

    // graph connectivity : store the index of neighbors in the graph's node list
    std::vector<size_t> parents;
    std::vector<size_t> children;

    // getters
    std::string name() const { return name_; }
    std::string op_type() const { return op_type_; }
    const std::vector<std::string> &inputs() const { return inputs_; }
    const std::vector<std::string> &outputs() const { return outputs_; }

    onnx::AttributeProto get_attribute(const std::string &key) const
    {
        if (attributes_.find(key) != attributes_.end())
        {
            return attributes_.at(key);
        }
        return onnx::AttributeProto(); // return empty if not found
    }

    void print() const
    {
        std::cout << "Node [" << op_type_ << "]: " << name_ << "\n";
        std::cout << "  In: " << parents.size() << " | Out: " << children.size() << " | Attrs: " << attributes_.size() << "\n";
    }

private:
    std::string name_{};
    std::string op_type_{};
    std::vector<std::string> inputs_{};
    std::vector<std::string> outputs_{};
    std::map<std::string, onnx::AttributeProto> attributes_;
};

#endif
