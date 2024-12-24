#ifndef NN_H
#define NN_H

#include "tensor4d.h"
#include <stdexcept>
#include <string>
#include <filesystem>
#include "cnpy/cnpy.h"

// Conv2D class definition
class Conv2D {
private:
    // Function to load kernel from the pretrained model
    Tensor4D loadKernelFromModel(const std::string& filename);

    // Function to load bias from the pretrained model
    Tensor4D loadBiasFromModel(const std::string& filename);

    // Function to shrink the matrix from the top left corner if dimensions are not suitable for max pooling
    static Tensor4D shrink_if_odd(const Tensor4D& input, int poolHeight, int poolWidth, int strideHeight, int strideWidth);

public:
    // Function to perform 2D convolution
    static Tensor4D conv2d(const Tensor4D& input, const Tensor4D& kernel, int stride = 1, int padding = 0, bool add_bias = false);

    // Function to perform 2D convolution by loading kernel from a file
    static Tensor4D conv2d(const Tensor4D& input, std::string filename, int stride = 1, int padding = 0, bool add_bias = false);

    // Function to upsample the tensor by a factor
    Tensor4D upsample(Tensor4D tensor, int scale = 2);

    // Function to perform max pooling
    static Tensor4D maxpool(const Tensor4D& input, int pooling = 2, int stride = 1, int padding = 0);
};

#endif // NN_H