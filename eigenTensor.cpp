#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include<chrono>


template <typename TensorType>
std::array<long, TensorType::NumIndices> getTensorShape(const TensorType& tensor) {
    std::array<long, TensorType::NumIndices> shape;
    for (int i = 0; i < TensorType::NumIndices; ++i) {
        shape[i] = tensor.dimension(i);
    }
    return shape;
}

int main() {
    // Initialize tensor (3, 128, 256, 256) [Batch, Channel, Height, Width]
    Eigen::Tensor<float, 4> input(1, 3, 256, 256);
    input.setRandom();

    // Initialize kernel (128, 128, 3, 3) [Output_Channel, Input_Channel, Height, Width]
    Eigen::Tensor<float, 4> kernel(128, 3, 3, 3);
    kernel.setRandom();


    int stride = 1;         // Stride for convolution
    int padding = 0;        // Padding for convolution
    bool add_bias = true;   // Option to add bias

    // Initialize bias if needed
    Eigen::Tensor<float, 1> bias(128); // One bias per output channel
    if (add_bias) {
        bias.setRandom();
    }

   

    int input_height = input.dimension(2); // Height of input tensor
    int input_width = input.dimension(3);  // Width of input tensor
    int kernel_height = kernel.dimension(2); // Kernel height
    int kernel_width = kernel.dimension(3);  // Kernel width

    int output_height = ((input_height - kernel_height + 2 * padding) / stride) + 1;
    int output_width = ((input_width - kernel_width + 2 * padding) / stride) + 1;



    // Initialize output tensor (Batch, Output_Channel, Height, Width)
    Eigen::Tensor<float, 4> output(1, kernel.dimension(0), output_height, output_width);
    output.setZero();
    
    auto output_shape1 = getTensorShape(output);
    std::cout << "Output Tensor Shape: [";
    for (size_t i = 0; i < output_shape1.size(); ++i) {
        std::cout << output_shape1[i] << (i < output_shape1.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    auto start_time0 = std::chrono::high_resolution_clock::now();

    // Perform convolution
    for (int b = 0; b < input.dimension(0); ++b) { // Batch
        for (int oc = 0; oc < kernel.dimension(0); ++oc) { // Output Channel
            for (int ic = 0; ic < kernel.dimension(1); ++ic) { // Input Channel
                for (int oh = 0; oh < output_height; ++oh) { // Output Height
                    for (int ow = 0; ow < output_width; ++ow) { // Output Width
                        float sum = 0.0f;
                        for (int kh = 0; kh < kernel.dimension(2); ++kh) { // Kernel Height
                            for (int kw = 0; kw < kernel.dimension(3); ++kw) { // Kernel Width
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && iw >= 0 && ih < input.dimension(2) && iw < input.dimension(3)) {
                                    sum += input(b, ic, ih, iw) * kernel(oc, ic, kh, kw);
                                }
                            }
                        }
                        output(b, oc, oh, ow) += sum;
                    }
                }
            }
            if (add_bias) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        output(b, oc, oh, ow) += bias(oc);
                    }
                }
            }
        }
    }
    std::cout << "Convolution Done" << std::endl;
	auto end_time0 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time0 - start_time0);
    std::cout << "Time taken for conv of input with 64x3x3: " << duration0.count() << " ms" << std::endl;


    // Print shape of output tensor
    auto output_shape = getTensorShape(output);
    std::cout << "Output Tensor Shape: [";
    for (size_t i = 0; i < output_shape.size(); ++i) {
        std::cout << output_shape[i] << (i < output_shape.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    return 0;
}
