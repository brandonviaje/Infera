#ifndef ATTRIBUTE_H
#define ATTRIBUTE_H

#include <string>
#include <variant>
#include <vector>
#include "onnx-ml.pb.h"

class Attribute
{
public:
    using AttributeValue = std::variant<int64_t, float, std::vector<int64_t>, std::string>;  // alias attribute value
 
    // constructor
    Attribute(const onnx::AttributeProto &attribute_proto) : name(attribute_proto.name())
    {
        switch (attribute_proto.type()) 
        {
        case onnx::AttributeProto::INT: 
        {
            value = attribute_proto.i();
            break;
        }
        case onnx::AttributeProto::FLOAT: 
        {
            value = attribute_proto.f();
            break;
        }
        case onnx::AttributeProto::INTS: 
        {
            std::vector<int64_t> ints(attribute_proto.ints_size());

            for (int i = 0; i < attribute_proto.ints_size(); ++i) 
            {
                ints[i] = attribute_proto.ints(i);
            }

            value = ints;
            break;
        }
        case onnx::AttributeProto::STRING:
        {
            value = attribute_proto.s();
            break;
        }
        default:
            throw std::runtime_error("Unsupported attribute type" + std::to_string(attribute_proto.type()));
        }
    }
    // getters
    const std::string &get_name() const { return name; }
    const AttributeValue &get_value() const { return value; }

private:
    std::string name{};
    AttributeValue value{};
};

#endif
