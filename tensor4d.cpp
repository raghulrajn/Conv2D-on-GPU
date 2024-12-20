#include<iostream>
#include<vector>
#include<Eigen/Dense>

class Tensor4D {
private:
    std::vector<std::vector<Eigen::MatrixXf>> tensor4d;
    int n; // dimensions (batch size)
    int c; // channels
    int h; // height
    int w; // width
    void checkDimensionsMatch(const Tensor4D& other) const {
        if (n != other.n || c != other.c || h != other.h || w != other.w) {
            throw std::invalid_argument("Tensor dimensions do not match for the operation");
        }
    }

public:
    // Constructor for 4D Tensor (N, C, H, W)
    Tensor4D(int dimensions, int channels, int height, int width)
        : n(dimensions), c(channels), h(height), w(width) {
        tensor4d.resize(n);  // Resize outer vector to hold 'n' batches
        for (int i = 0; i < n; ++i) {
            tensor4d[i].resize(c);  // Resize each batch to hold 'c' channels
            for (int j = 0; j < c; ++j) {
                tensor4d[i][j] = Eigen::MatrixXf::Zero(h, w);  // Initialize each matrix to zeros
            }
        }
    }

    // Constructor for 4D Tensor with 1 Dimension (1,C, H, W)
    Tensor4D(int channels, int height, int width)
        : Tensor4D(1, channels, height, width) {}

    //shape method to return the shape of the tensor as tuple(n,c,h,w)
    std::tuple<int, int, int, int> shape() const {
        return std::make_tuple(this->n, this->c, this->h, this->w);
    }

    //Resize the matrix inside the tensor i.e to modify the h and w of a tensor. Modifies the shape in place and set the value to zero
    void resizeMatrices(int new_height, int new_width) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                tensor4d[i][j].conservativeResize(new_height, new_width);  // Resize while preserving existing data
                tensor4d[i][j]= Eigen::MatrixXf::Zero(new_height, new_width);
            }
        }
    }

    //Resize the tensor dimension and channels i.e to modify the n and c of a tensor.
    // void resizeTensor(int new_dim, int new_channel) {
    //     for (int i = 0; i < new_dim; ++i) {
    //         for (int j = 0; j < new_channel; ++j) {
    //             tensor4d[i][j].conservativeResize(h, w);  // Resize while preserving existing data
    //         }
    //     }
    // }

    // Accessor for the tensor data
    Eigen::MatrixXf& operator()(int i, int j) {
        return tensor4d[i][j];
    }

    const Eigen::MatrixXf& operator()(int i, int j) const {
        return tensor4d[i][j];
    }

    // Display function to print the contents of the tensor
    void print() const {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                std::cout << "Tensor4D[" << i << "][" << j << "]:\n";
                std::cout << tensor4d[i][j] << "\n\n";
            }
        }
    }

    // Method to get each dimension of the tensor
    size_t dimension(size_t index) const {
    switch(index) {
        case 0: return n; //dimension
        case 1: return c; //channel
        case 2: return h; //height
        case 3: return w; //width
        default: throw std::out_of_range("Invalid dimension index");
         }
    }

    // Overload the addition operator (Tensor4D + Tensor4D)
    Tensor4D operator+(const Tensor4D& other) const {
        checkDimensionsMatch(other);
        Tensor4D result(n, c, h, w);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                result.tensor4d[i][j] = tensor4d[i][j] + other.tensor4d[i][j];
            }
        }
        return result;
    }

    // Overload the subtraction operator (Tensor4D - Tensor4D)
    Tensor4D operator-(const Tensor4D& other) const {
        checkDimensionsMatch(other);
        Tensor4D result(n, c, h, w);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                result.tensor4d[i][j] = tensor4d[i][j] - other.tensor4d[i][j];
            }
        }
        return result;
    }

    // Overload the elementwise multiplication operator (Tensor4D * Tensor4D)
    Tensor4D operator*(const Tensor4D& other) const {
        checkDimensionsMatch(other);
        Tensor4D result(n, c, h, w);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                result.tensor4d[i][j] = tensor4d[i][j].cwiseProduct(other.tensor4d[i][j]);
            }
        }
        return result;
    }

    // Overload the scalar addition operator (Tensor4D + scalar)
    Tensor4D operator+(float scalar) const {
        Tensor4D result(n, c, h, w);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                result.tensor4d[i][j] = tensor4d[i][j] * scalar;
            }
        }
        return result;
    }

    // Overload the scalar subtraction operator (Tensor4D - scalar)
    Tensor4D operator-(float scalar) const {
        Tensor4D result(n, c, h, w);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                result.tensor4d[i][j] = tensor4d[i][j] * scalar;
            }
        }
        return result;
    }

    // Overload the scalar multiplication operator (Tensor4D * scalar)
    Tensor4D operator*(float scalar) const {
        Tensor4D result(n, c, h, w);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                result.tensor4d[i][j] = tensor4d[i][j] * scalar;
            }
        }
        return result;
    }

    // Overload the scalar division operator (Tensor4D / scalar)
    Tensor4D operator/(float scalar) const {
        Tensor4D result(n, c, h, w);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                result.tensor4d[i][j] = tensor4d[i][j] / scalar;
            }
        }
        return result;
    }

    //Applies relu activation for the tensor4D object inplace
    void relu() {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                tensor4d[i][j] = tensor4d[i][j].cwiseMax(0.0f);
            }
        }
    }

    //Applies Batch normalisation to the tensor4D object inplace
    void batch_normalize(float epsilon = 1e-5) {
        // Initialize gamma and beta if they are not provided
        Eigen::VectorXf gamma, beta;
        if (gamma.size() == 0) gamma = Eigen::VectorXf::Ones(c);
        if (beta.size() == 0) beta = Eigen::VectorXf::Zero(c);

        //Normalize values for the all the features (channels) after convolution operation
        for (int channel = 0; channel < c; ++channel) {
            // Calculate mean and variance across the batch for each channel
            Eigen::MatrixXf sum = Eigen::MatrixXf::Zero(h, w);
            Eigen::MatrixXf sum_sq = Eigen::MatrixXf::Zero(h, w);

            // Accumulate sums
            for (int batch = 0; batch < n; ++batch) {
                sum += tensor4d[batch][channel];
                sum_sq += tensor4d[batch][channel].array().square().matrix();
            }

            Eigen::MatrixXf mean = sum / n;
            Eigen::MatrixXf variance = (sum_sq / n) - mean.array().square().matrix();

            // Normalize each batch and apply gamma and beta
            for (int batch = 0; batch < n; ++batch) {
                tensor4d[batch][channel] = ((tensor4d[batch][channel] - mean).array() / (variance.array().sqrt() + epsilon)).matrix();
                tensor4d[batch][channel] = (gamma(channel) * tensor4d[batch][channel]).array() + beta(channel);
            }
        }
    }

    void printShape(){
        std::cout<<"("<<n<<","<<c<<","<<h<<","<<w<<")"<<std::endl;
    }
};

// int main() {
//     int N = 1;  // Batch size
//     int C = 1;  // Channels
//     int H = 3;  // Height
//     int W = 3;  // Width

//     Tensor4D input_tensor1(N, C, H, W);
//     Tensor4D input_tensor2(N, C, H, W);

//     // Initialize one of the matrices with random values
//     input_tensor1(0, 0) = Eigen::MatrixXf::Random(H, W);
//     input_tensor2(0, 0) = Eigen::MatrixXf::Random(H, W);

//     input_tensor2(0, 0)(0,0) = 100;
//     input_tensor2(0, 0)(0,1) = 200;

//     Tensor4D add  = input_tensor1 + input_tensor2;

//     std::cout << "Initialized Tensor4D:\n";
//     std::cout << add.dimension(0)<<"\n";
//     std::cout << add.dimension(1)<<"\n";
//     std::cout << add.dimension(2)<<"\n";
//     std::cout << add.dimension(3)<<"\n";
//     add.print();

//     return 0;
// }