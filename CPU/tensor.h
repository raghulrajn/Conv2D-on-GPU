#ifndef TENSOR4D_H
#define TENSOR4D_H

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

class Tensor4D {

public:
    Eigen::Tensor<float, 4> tensor;
    int n; // dimensions (batch size)
    int c; // channels
    int h; // height
    int w; // width
    // Constructors
    Tensor4D(int N, int C, int H, int W);
    Tensor4D(int C, int H, int W);
    Tensor4D(int H, int W);
    Tensor4D(int N, int C, int H, int W, Eigen::Tensor<float, 4> existing_tensor);

    // Accessor for getting a matrix of shape (H, W)
    Eigen::Tensor<float, 2> get_matrix(int n, int c);

    // Padding functions
    Tensor4D pad(int padH, int padW);
    void addPadding(int padH, int padW);
    void addBias(Tensor4D& bias);

    // Print shape of the tensor
    void print_shape() const;

    // Get each dimension of the tensor
    int dimension(int index) const;

    // Element-wise tensor operations
    Tensor4D operator+(const Tensor4D& other) const;
    Tensor4D operator-(const Tensor4D& other) const;
    Tensor4D operator*(const Tensor4D& other) const;
    Tensor4D operator/(const Tensor4D& other) const;

    // Scalar operations on tensor
    Tensor4D operator+(float scalar) const;
    Tensor4D operator-(float scalar) const;
    Tensor4D operator*(float scalar) const;
    Tensor4D operator/(float scalar) const;

    // ReLU activation function
    void relu();

    // Print each channel of the tensor as a 2D matrix
    void printTensor() const;

    // Concatenation functions
    static Tensor4D concatenate(const Tensor4D& t1, const Tensor4D& t2);
    Tensor4D concatenate(const Tensor4D& other) const;

    // Function to get a block (H, W) from given (r, c)
    Eigen::Tensor<float, 2> block(int n, int c, int r, int col, int height, int width);
};

#endif // TENSOR4D_H