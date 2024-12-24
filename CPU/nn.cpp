#include "tensor4d.cpp"
#include <stdexcept>
#include <string>
#include <filesystem>
#include "cnpy/cnpy.h"
#include <iostream>

class Conv2D{

    private:
    // Loading kernel from the pretrained model and assigining the kernel to as tensor4D object
    Tensor4D loadKernelFromModel(const std::string& filename) {
        std::string filepath = "./kernels/" + filename + "_weight.npy";
        cnpy::NpyArray arr = cnpy::npy_load(filepath);

        // Check that the data type is float
        if (arr.word_size != sizeof(float)) {
            throw std::runtime_error("Data type mismatch: expected float data type.");
        }

        if (arr.shape.size() == 1) {
            // For 1D array (e.g., 128, which could be interpreted as (1, 1, 1, 128))
            n = 1;
            c = 1;
            h = 1;
            w = arr.shape[0];
        } else if (arr.shape.size() == 2) {
            // For 2D array, assuming it is a (1, 1, 64, 64) shape
            n = 1;
            c = 1;
            h = arr.shape[0];
            w = arr.shape[1];
        } else if (arr.shape.size() == 4) {
            // For 4D array, it is in the format (n, c, h, w)
            n = arr.shape[0];
            c = arr.shape[1];
            h = arr.shape[2];
            w = arr.shape[3];
        } else {
            throw std::runtime_error("Unsupported kernel shape.");
        }

        // Create the Tensor4D object
        Tensor4D kernel(N, C, H, W);

        // Copy data from the NpyArray to the Tensor4D object
        float* data = arr.data<float>();
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        kernel.tensor(n, c, h, w) = data[n * C * H * W + c * H * W + h * W + w];
                    }
                }
            }
        }

        return kernel;
    }

    // Loading Bias from the pretrained model and assigining the kernel to as tensor4D object
    Tensor4D loadBiasFromModel(const std::string& filename) {
        std::string filepath = "./kernels/" + filename + "_bias.npy";
        cnpy::NpyArray arr = cnpy::npy_load(filepath);

        // Check that the data type is float
        if (arr.word_size != sizeof(float)) {
            throw std::runtime_error("Data type mismatch: expected float data type.");
        }

        if (arr.shape.size() == 1) {
            // For 1D array (e.g., 128, which could be interpreted as (1, 1, 1, 128))
            n = 1;
            c = 1;
            h = 1;
            w = arr.shape[0];
        } 
        else {
            throw std::runtime_error("Unsupported kernel shape.");
        }

        // Create the Tensor4D object
        Tensor4D bias(N, C, H, W);

        // Copy data from the NpyArray to the Tensor4D object
        float* data = arr.data<float>();
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        bias.tensor(n, c, h, w) = data[n * C * H * W + c * H * W + h * W + w];
                    }
                }
            }
        }

        return bias;
    }

    // Shrink the matrix from the top left cornner if the padding is >0 and the matrix is having odd dimensions
    static Tensor4D shrink_if_odd(const Tensor4D& input, int poolHeight, int poolWidth, int strideHeight, int strideWidth) {
        int new_h = (input.h - poolHeight) % strideHeight == 0 ? input.h : input.h - 1;
        int new_w = (input.w - poolWidth) % strideWidth == 0 ? input.w : input.w - 1;

        if (new_h != input.h || new_w != input.w) {
            Eigen::array<int, 4> offsets = {0, 0, 0, 0};
            Eigen::array<int, 4> extents = {input.n, input.c, new_h, new_w};
            Tensor4D result(input.n, input.c, new_h, new_w);
            result.tensor = input.tensor.slice(offsets, extents);
            return result;
        }
        return input; // Return the original tensor if no shrinking is needed
    }

    public:

    static Tensor4D conv2d(const Tensor4D& input, const Tensor4D& kernel, int stride=1, int padding=0, bool add_bias = false) {
        int output_height = (input.h - kernel.h + 2 * padding) / stride + 1;
        int output_width = (input.w - kernel.w + 2 * padding) / stride + 1;
        Tensor4D output(input.n, kernel.n, output_height, output_width);

        if(padding>0){
            input.addPadding(padding, padding);
        }

        for (int b = 0; b < input.n; ++b) { // Batch
            for (int oc = 0; oc < kernel.n; ++oc) { // Output Channel
                for (int ic = 0; ic < kernel.c; ++ic) { // Input Channel
                    for (int oh = 0; oh < output_height; ++oh) { // Output Height
                        for (int ow = 0; ow < output_width; ++ow) { // Output Width
                            float sum = 0.0f;
                            for (int kh = 0; kh < kernel.h; ++kh) { // Kernel Height
                                for (int kw = 0; kw < kernel.w; ++kw) { // Kernel Width
                                    int ih = oh * stride - padding + kh;
                                    int iw = ow * stride - padding + kw;
                                    if (ih >= 0 && iw >= 0 && ih < input.h && iw < input.w) {
                                        sum += input.tensor(b, ic, ih, iw) * kernel.tensor(oc, ic, kh, kw);
                                    }
                                }
                            }
                            output.tensor(b, oc, oh, ow) += sum;
                        }
                    }
                }
                if (add_bias) {
                    for (int oh = 0; oh < output_height; ++oh) {
                        for (int ow = 0; ow < output_width; ++ow) {
                            output.tensor(b, oc, oh, ow) += bias(oc);
                        }
                    }
                }
            }
        }
        return output;
    }

    static Tensor4D conv2d(const Tensor4D& input, std::string filename, int stride=1, int padding=0, bool add_bias = false) {
        
        Tensor4D kernel = loadKernelFromModel(filename);
        
        
        int output_height = (input.h - kernel.h + 2 * padding) / stride + 1;
        int output_width = (input.w - kernel.w + 2 * padding) / stride + 1;
        Tensor4D output(input.n, kernel.n, output_height, output_width);

        if(padding>0){
            input.addPadding(padding, padding);
        }

        for (int b = 0; b < input.n; ++b) { // Batch
            for (int oc = 0; oc < kernel.n; ++oc) { // Output Channel
                for (int ic = 0; ic < kernel.c; ++ic) { // Input Channel
                    for (int oh = 0; oh < output_height; ++oh) { // Output Height
                        for (int ow = 0; ow < output_width; ++ow) { // Output Width
                            float sum = 0.0f;
                            for (int kh = 0; kh < kernel.h; ++kh) { // Kernel Height
                                for (int kw = 0; kw < kernel.w; ++kw) { // Kernel Width
                                    int ih = oh * stride - padding + kh;
                                    int iw = ow * stride - padding + kw;
                                    if (ih >= 0 && iw >= 0 && ih < input.h && iw < input.w) {
                                        sum += input.tensor(b, ic, ih, iw) * kernel.tensor(oc, ic, kh, kw);
                                    }
                                }
                            }
                            output.tensor(b, oc, oh, ow) += sum;
                        }
                    }
                }
                if (add_bias) {
                    for (int oh = 0; oh < output_height; ++oh) {
                        for (int ow = 0; ow < output_width; ++ow) {
                            output.tensor(b, oc, oh, ow) += bias(oc);
                        }
                    }
                }
            }
        }
        return output;
    }
     // Function to upsample the tensor by a factor of 2x2
    Tensor4D upsample(Tensor4D tensor, int scale=2) {
        int newH = tensor.dimension(2) * scale;
        int newW = tensor.dimension(3) * scale;
        Tensor4D result(tensor.dimension(0), tensor.dimension(1), newH, newW);

         for (int b = 0; b < n; ++b) {
            for (int ch = 0; ch < c; ++ch) {
                for (int y = 0; y < newH; ++y) {
                    for (int x = 0; x < newW; ++x) {
                        // Compute the coordinates of the 4 surrounding pixels
                        float srcY = (float)(y) / 2.0f;
                        float srcX = (float)(x) / 2.0f;
                        int y0 = (int)std::floor(srcY);
                        int x0 = (int)std::floor(srcX);
                        int y1 = std::min(y0 + 1, h - 1);
                        int x1 = std::min(x0 + 1, w - 1);

                        // Compute the interpolation weights
                        float wy1 = srcY - y0;
                        float wy0 = 1.0f - wy1;
                        float wx1 = srcX - x0;
                        float wx0 = 1.0f - wx1;

                        // Perform the bilinear interpolation
                        result.tensor(b, ch, y, x) = 
                            wy0 * (wx0 * tensor(b, ch, y0, x0) + wx1 * tensor(b, ch, y0, x1)) +
                            wy1 * (wx0 * tensor(b, ch, y1, x0) + wx1 * tensor(b, ch, y1, x1));
                    }
                }
            }
        }
        
        return result;
    }

    // Max pooling function with stride and padding options
    static Tensor4D maxpool(const Tensor4D& input, int pooling = 2, h, int stride=1, int padding=0) {
        // Add padding
        int paddedH = input.h + 2 * padding;
        int paddedW = input.w + 2 * padding;
        Tensor4D paddedTensor(input.n, input.c, paddedH, paddedW);
        paddedTensor.tensor.setZero();
        paddedTensor.tensor.slice(Eigen::array<int, 4>({0, 0, padding, padding}), Eigen::array<int, 4>({input.n, input.c, input.h, input.w})) = input.tensor;

        // Ensure the dimensions are even and compatible with pooling
        Tensor4D shrunkenTensor = shrink_if_odd(paddedTensor, pooling, pooling, stride, stride);

        // Calculate new dimensions
        int newH = (shrunkenTensor.h - pooling) / stride + 1;
        int newW = (shrunkenTensor.w - pooling) / stride + 1;

        // Create new tensor for the result
        Tensor4D result(shrunkenTensor.n, shrunkenTensor.c, newH, newW);

        // Perform max pooling
        for (int n = 0; n < shrunkenTensor.n; ++n) {
            for (int c = 0; c < shrunkenTensor.c; ++c) {
                for (int h = 0; h < newH; ++h) {
                    for (int w = 0; w < newW; ++w) {
                        result.tensor(n, c, h, w) = shrunkenTensor.tensor.chip(n, 0).chip(c, 0)
                            .slice(Eigen::array<int, 2>({h * stride, w * stride}), Eigen::array<int, 2>({pooling, pooling}))
                            .maximum();
                    }
                }
            }
        }
        return result;
    }

};