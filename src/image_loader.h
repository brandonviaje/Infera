#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <string>
#include <vector>
#include "tensor.h"

class ImageLoader 
{
public:
    static Tensor<float>* load_image(const std::string& filepath, int target_w = 0, int target_h = 0);
};

#endif
