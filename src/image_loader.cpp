#include "image_loader.h"
#include <iostream>
#include <vector>
#include <stdexcept>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-value"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

Tensor<float>* ImageLoader::load_image(const std::string& filepath, int target_w, int target_h) 
{
    int width, height, channels;
    
    // load image
    unsigned char* img_data = stbi_load(filepath.c_str(), &width, &height, &channels, 1);

    if (!img_data) 
    {
        throw std::runtime_error("Failed to load image: " + filepath + " (" + stbi_failure_reason() + ")");
    }

    unsigned char* process_img = img_data;
    std::vector<unsigned char> resized_buffer;
    int final_w {width};
    int final_h {height};

    // handle resizing
    if (width != target_w || height != target_h)
    {
        std::cout << "Resizing image: " << width << "x" << height << " -> " << target_w << "x" << target_h << "\n";
        
        resized_buffer.resize(target_w * target_h);
        stbir_resize_uint8(img_data, width, height, 0,  resized_buffer.data(), target_w, target_h, 0, 1);
        
        process_img = resized_buffer.data();
        final_w = target_w;
        final_h = target_h;
    }

    // create tensor
    auto* tensor = new Tensor<float>({1, 1, (size_t)final_h, (size_t)final_w});
    float* tensor_data = tensor->data();

    // normalize
    int total_pixels = final_w * final_h;
    for (int i = 0; i < total_pixels; ++i) 
    {
        tensor_data[i] = static_cast<float>(process_img[i]) / 255.0f;
    }

    stbi_image_free(img_data);
    
    return tensor;
}
