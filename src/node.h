#ifndef NODE_H
#define NODE_H

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "attribute.h"
#include "onnx-ml.pb.h"

class Node
{
public:
    Node(const onnx::NodeProto &node_proto) : name_(node_proto.name()), optype_(node_proto.op_type()),inputs_(node_proto.input().begin(), node_proto.input().end()), outputs_(node_proto.output().begin(), node_proto.output().end()) 
    { 
        // parse attributes from node 
        for (const auto &attribute_proto : node_proto.attribute()) 
        {
            attributes_.emplace(attribute_proto.name(), Attribute(attribute_proto)); // key: attr name, val: attr object
        }
    }

    // getters and setters
    std::string get_name() const { return name_; }
    std::string get_optype() const { return optype_; }
    const std::vector<std::string> &get_inputs() const { return inputs_; }
    const std::vector<std::string> &get_outputs() const { return outputs_; }
    const std::unordered_map<std::string, Attribute> &get_attributes() const {return attributes_;}
    void add_inputs(std::string input) { inputs_.push_back(input);}
    void add_outputs(std::string output) {outputs_.push_back(output);};
    void set_name(const std::string& name) { name_ = name; }
    
    template <typename T>
    std::optional<T> get_attribute(const std::string &name) const
    {
        // check if attribute exists and is of type T
        if(attributes_.find(name) != attributes_.end() && std::holds_alternative<T>(attributes_.find(name)->second.get_value()))
        {
            return std::get<T>(attributes_.find(name)->second.get_value());    // return value
        }
        return std::nullopt;                                                   // else return nullopt
    }
    
private:
    std::string name_{};
    std::string optype_{};
    std::vector<std::string> inputs_{};
    std::vector<std::string> outputs_{};
    std::unordered_map<std::string, Attribute> attributes_;
};

#endif
