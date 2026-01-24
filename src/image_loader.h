#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "tensor.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class ImageLoader 
{
public:
    static Tensor<float>* load_image(const std::string& filepath) 
    {
        int width, height, channels;
        
        // load the Image
        unsigned char* img_data = stbi_load(filepath.c_str(), &width, &height, &channels, 1);

        if (!img_data) 
        {
            throw std::runtime_error("Failed to load image: " + filepath + " (" + stbi_failure_reason() + ")");
        }

   
        std::cout << "Processing image: " << filepath << " [" << width << "x" << height << "]\n";

        // cast int -> std::size_t for shape vector
        std::size_t h = static_cast<std::size_t>(height);
        std::size_t w = static_cast<std::size_t>(width);

        // shape: [Batch=1, Channels=1, Height, Width]
        auto* tensor = new Tensor<float>({1, 1, h, w});
        float* tensor_data = tensor->data();

        // normalize and copy
        int total_pixels {width * height};

        for (int i = 0; i < total_pixels; ++i) 
        {
            tensor_data[i] = static_cast<float>(img_data[i]) / 255.0f;
        }

        stbi_image_free(img_data);
        
        return tensor;
    }
};

#endif
