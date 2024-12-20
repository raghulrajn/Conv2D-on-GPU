#include <iostream>
#include <sstream>
#include <vector>
#include "tensor4d.cpp"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <filesystem>
#include "cnpy/cnpy.h"
// g++ -std=c++17 -I/cnpy -I/usr/include/eigen3 -L/cnpy/build -o conv conv2D.cpp kernel_loader.cpp cnpy/build/libcnpy.a -lz
class Conv2D{

    private:
        bool checkDimension(Tensor4D t){
            int h = t.dimension(2);
            int w = t.dimension(3);
            if((h%2 == 0 && w%2 ==0) && h == w) return true;

            else if((h%2 != 0 || w%2 !=0)) return false;

            else if (h !=w) return false;

            else return false;
        }

        Tensor4D padTensor(Tensor4D inputTensor, int padding){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h = inputTensor.dimension(2);
            int w = inputTensor.dimension(3);

            int h_out = h+padding*2;
            int w_out = w+padding*2;

            Tensor4D outputTensor(dim, channels, h_out, w_out);

            const int h_in = h;
            const int w_in = w;
            
            //  Eigen::MatrixXf newMatrix = Eigen::MatrixXd::Zero(h+padding*2, w+padding*2);
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < channels; j++){
                    Eigen::MatrixXf newMatrix = Eigen::MatrixXf::Zero(h+padding*2, w+padding*2);
                    newMatrix.block(padding, padding, h,w) = inputTensor(i,j);
                    outputTensor(i,j) = newMatrix;
                    // newMatrix.block(padding, padding, h,w) = inputTensor(i,j);
                }
            }


            return outputTensor;
        }

        // Shrink the matrix from the top left cornner if the padding is >0 and the matrix is having odd dimensions
        Tensor4D shrinkMatrix(Tensor4D inputTensor){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h = inputTensor.dimension(2);
            int w = inputTensor.dimension(3);
            std::cout<<h<<"  "<<w<<std::endl;
            if (h==w && h%2 != 0 && w%2!=0){
                h = h-1;
                w = w-1;
            }
            else{
                int minEven = std::min(h,w) & ~1;
                h = minEven;
                w = minEven;
            }
            assert(h%2==0);
            assert(w%2==0);
            Tensor4D outputTensor(dim, channels, h, w);

            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < channels; j++){
                    outputTensor(i,j) = inputTensor(i,j);
                    }
                }
            return outputTensor;
        }
        
        // Kernel initilization using kaiming for fixed kernel size (number of filters, kernel size)
        std::vector<Eigen::MatrixXf> kernelInitialization(int dim, int kernel_size){
            
            std::vector<Eigen::MatrixXf> filters;
            filters.resize(dim);
            // std::cout<<"resized"<<"\n";
            for(int i=0; i<dim;++i)
            {
                float std = sqrt(2/(float) kernel_size);
                std::random_device rd;
                std::mt19937 gen(rd()); 
                std::normal_distribution<float> dist(0.0f, std); 
                Eigen::MatrixXf fil(3,3);
                for (int col = 0; col < kernel_size; col++) {
                    for (int row = 0; row < kernel_size; row++) {
                        // fil(row, col) = dist(gen);
                        fil(row, col) = 1;
                    }
                }
                // std::cout<<"i = "<<i<<"\n";
                filters[i] = fil;
            }
            // std::cout<<"out of for loop"<<"\n";
            return filters;
        }

        // Loading kernel from the pretrained model and assigining the kernel to as tensor4D object
        Tensor4D loadKernelFromModel(const std::string& filename){
            std::string filepath = "./kernels/"+filename+"_weight.npy";
            cnpy::NpyArray arr = cnpy::npy_load(filepath);
            int n,c,h,w;
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
            //load the kernel from npy file into Tensor4D object manually
            Tensor4D kernel(n,c,h,w);
            float* data = arr.data<float>();
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < c; ++j) {
                    for (int x = 0; x < h; ++x) {
                        for (int y = 0; y < w; ++y) {
                            kernel(i,j)(x, y) = data[(i * c * h * w) + (j * h * w) + (x * w) + y];
                        }
                    }
                }
            }
            return kernel;
        }

        Tensor4D loadBiasFromModel(const std::string& filename){
            std::string filepath = "./kernels/"+filename+"_bias.npy";
            cnpy::NpyArray arr = cnpy::npy_load(filepath);
            int n,c,h,w;
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
            //load the kernel from npy file into Tensor4D object manually
            Tensor4D kernel(n,c,h,w);
            float* data = arr.data<float>();
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < c; ++j) {
                    for (int x = 0; x < h; ++x) {
                        for (int y = 0; y < w; ++y) {
                            kernel(i,j)(x, y) = data[(i * c * h * w) + (j * h * w) + (x * w) + y];
                        }
                    }
                }
            }
            return kernel;
        }

        Tensor4D addBias(const Tensor4D &inputTensor, const Tensor4D &bias) {
        int dim = bias.dimension(0);
        int channels = bias.dimension(1);
        int h = bias.dimension(2);
        int w = bias.dimension(3);
        // Ensure the bias tensor has the correct shape: (1, 1, 1, numFilters)
        if (dim != 1 || channels != 1 || h != 1 || w != inputTensor.dimension(0)) {
            throw std::invalid_argument("Bias dimensions do not match the output tensor's number of filters.");
        }

        Tensor4D result = inputTensor;

        // Iterate over each batch and each output channel
        for (int b = 0; b < result.dimension(0); ++b) {
            for (int f = 0; f < result.dimension(1); ++f) {
                // Get the bias value from tensor
                float biasValue = bias(0,0)(0, f);
                // Add the bias to every element in the output matrix
                result(b,f).array() += biasValue;
            }
        }
        return result;
    }
        
        template <typename matrixBlock>
        double dot(matrixBlock m1, Eigen::MatrixXf m2){
            double result=0;
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    result += m1(i,j)*m2(i,j);
                }
            }
            return result;
        }
        
        
    public:
        Tensor4D conv2d(Tensor4D &inputTensor,int outputdim, const int kernel_size=3, int stride=1, int padding=0, bool bias=false){
            int dim      = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h_in     = inputTensor.dimension(2);
            int w_in     = inputTensor.dimension(3);
            const int k_z = kernel_size; 

            int h_out = floor(((h_in + 2*padding - (kernel_size-1)-1)/stride) +1);
            int w_out = floor(((w_in + 2*padding - (kernel_size-1)-1)/stride) +1);
            int row_stride = 0;
            int col_stride = 0;
            
            if(padding>0){
                inputTensor = padTensor(inputTensor, padding);
            }
            std::vector<Eigen::MatrixXf> filters = kernelInitialization(outputdim, kernel_size);
            
            Tensor4D outputTensor(dim, channels, h_out, w_out);

            if(checkDimension(inputTensor))
            {
                for (int i = 0; i < dim; i++) 
                {
                    // std::cout<<"dim = "<<i<<"\n";
                    int rows = 0;
                    int cols = 0;
                    for (int j = 0; j < channels; j++) 
                    {
                    // std::cout<<"channel = "<<j<<"\n";
                    while(rows<h_in - (kernel_size-1))
                    {
                        while(cols<w_in-(kernel_size-1))
                        {
                            // std::cout<<"rows = "<<rows<<" cols = "<<cols<<"\n";
                            double d_p = dot(inputTensor(i,j).block<3,3>(rows,cols), filters[dim]);
                            cols+=stride;
                            outputTensor(i, j)(row_stride,col_stride) = d_p;
                            col_stride++;
                        }
                        rows+=stride;
                        cols=0;
                        row_stride++;
                        col_stride = 0;
                    }
                    }
                }

            }

        return outputTensor;
        }

        Tensor4D conv2d(Tensor4D &inputTensor, std::string filename, bool bias=false){
            int dim      = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h_in     = inputTensor.dimension(2);
            int w_in     = inputTensor.dimension(3);
            int kernel_size = 3;
            int stride = 1;

            Tensor4D filters = loadKernelFromModel(filename);
            filters.printShape();
            
            int out_channels = filters.dimension(0);
            int in_channels  = filters.dimension(1);
            int ker_h        = filters.dimension(2);
            int ker_w        = filters.dimension(3);

            assert(("Input tensor channels and Filter channels are not matching", channels==in_channels));

            int row_stride = 0;
            int col_stride = 0;

            int out_h = (h_in - ker_h) / stride + 1;
            int out_w = (w_in - ker_w) / stride + 1;
            
            Tensor4D outputTensor(dim, out_channels, out_h, out_w);

            // for (int i = 0; i < dim; i++) {
            //     int rows = 0;
            //     int cols = 0;
            //     for (int j = 0; j < out_channels; j++) {
            //         while(rows<h_in - (kernel_size-1)){
            //             while(cols<w_in-(kernel_size-1))
            //             {
            //                 double d_p = dot(inputTensor(i,j).block<3,3>(rows,cols), filters(i,j));
            //                 cols+=stride;
            //                 outputTensor(i, j)(row_stride,col_stride) = d_p;
            //                 col_stride++;
            //             }
            //             rows+=stride;
            //             cols=0;
            //             row_stride++;
            //             col_stride = 0;
            //     }
            //     }
            // }

            for (int b = 0; b < dim; b++) {  // For each batch
                for (int f = 0; f < out_channels; f+=1) {  // For each filter (output channel)
                    Eigen::MatrixXf result = Eigen::MatrixXf::Zero(out_h, out_w);
                    for (int i = 0; i <= h_in - ker_h; i += stride) {
                        for (int j = 0; j <= w_in - ker_w; j += stride) {
                            float sum = 0.0;
                            for (int ch = 0; ch < in_channels; ++ch) {  // For each input channel
                                // std::cout<<b<<" "<<f<<" "<<i <<" "<<j<<" "<<ch<<std::endl;
                                Eigen::MatrixXf inputRegion = inputTensor(b,ch).block(i, j, ker_h, ker_w);
                                // std::cout<<b<<" "<<f<<std::endl;
                                sum+=dot(inputRegion, filters(f,ch));
                                // sum += (inputRegion.array() * filters(b,f).array()).sum();
                            }
                            // std::cout<<b<<" "<<f<<" "<<i <<" "<<j<<" "<<std::endl;
                            result(i / stride, j / stride) = sum;
                        }
                    }
                    outputTensor(b,f) = result;
                }
            }

            if(bias){
                Tensor4D bias = loadBiasFromModel(filename);
                outputTensor = addBias(outputTensor, bias);
            }

        return outputTensor;
        }

        Tensor4D maxpool2d(Tensor4D &inputTensor, int kernel_size=2, int stride=2, int padding=0, bool ceil_mode=false){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            const int conv_h = inputTensor.dimension(2);
            const int conv_w = inputTensor.dimension(3);
            if (padding>(int)kernel_size/2){
                throw std::invalid_argument( "padding should be at most half of effective kernel size");
            }

            if(padding>0){
                inputTensor = padTensor(inputTensor, padding);
            }

            inputTensor = shrinkMatrix(inputTensor);

            int h_in     = inputTensor.dimension(2);
            int w_in     = inputTensor.dimension(3);

            int h_out = floor(((conv_h + 2*padding - (kernel_size-1)-1)/stride) +1);
            int w_out = floor(((conv_w + 2*padding - (kernel_size-1)-1)/stride) +1);
            int row_stride = 0;
            int col_stride = 0;
            
            Tensor4D outputTensor(dim, channels, h_out, w_out);

            for (int i = 0; i < dim; i++) 
            {
                // std::cout<<"dim = "<<i<<"\n";
                int rows = 0;
                int cols = 0;
                for (int j = 0; j < channels; j++) 
                {
                // std::cout<<"channel = "<<j<<"\n";
                while(rows<h_in - (kernel_size-1))
                {
                    while(cols<w_in-(kernel_size-1))
                    {
                        float d_p = inputTensor(i,j).block<2,2>(rows,cols).maxCoeff();
                        cols+=stride;
                        outputTensor(i, j)(row_stride,col_stride) = d_p;
                        col_stride++;
                    }
                    rows+=stride;
                    cols=0;
                    row_stride++;
                    col_stride = 0;
                }
                }
            }
        return outputTensor;
        }

        Tensor4D upsampling(Tensor4D &inputTensor, int scale_factor=2, std::string mode="bilinear"){
            int dim      = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h_in     = inputTensor.dimension(2);
            int w_in     = inputTensor.dimension(3);

            int new_h = static_cast<int>(h_in * scale_factor);
            int new_w = static_cast<int>(w_in * scale_factor);
            Tensor4D upsampled_tensor(dim, channels, h_in*scale_factor, w_in*scale_factor);

        // Perform bilinear interpolation on each channel and batch
            for (int i = 0; i < dim; ++i) {  
                for (int j = 0; j < channels; ++j) {
                    Eigen::MatrixXf& input = inputTensor(i,j);
                    Eigen::MatrixXf& output = upsampled_tensor(i,j);

                    // Loop over the new (upsampled) size
                    for (int i = 0; i < new_h; ++i) {
                        for (int j = 0; j < new_w; ++j) {
                            // Find the corresponding position in the original image
                            float srcRow = i / scale_factor;
                            float srcCol = j / scale_factor;

                            // Get the four neighboring pixels
                            int r0 = std::floor(srcRow);  // Top row index
                            int c0 = std::floor(srcCol);  // Left column index
                            int r1 = std::min(r0 + 1, h_in - 1); // Bottom row index
                            int c1 = std::min(c0 + 1, w_in - 1); // Right column index

                            float dr = srcRow - r0;
                            float dc = srcCol - c0;

                            // Bilinear interpolation formula
                            output(i, j) = (1 - dr) * (1 - dc) * input(r0, c0) +
                                        (1 - dr) * dc * input(r0, c1) +
                                        dr * (1 - dc) * input(r1, c0) +
                                        dr * dc * input(r1, c1);
                        }
                    }
                }
            }

        return upsampled_tensor;
    }

        Tensor4D concatenate(Tensor4D &encoder, Tensor4D &decoder){
            int enc_dim      = encoder.dimension(0);
            int enc_channels = encoder.dimension(1);
            int enc_h        = encoder.dimension(2);
            int enc_w        = encoder.dimension(3);

            int dec_dim      = decoder.dimension(0);
            int dec_channels = decoder.dimension(1);
            int dec_h        = decoder.dimension(2);
            int dec_w        = decoder.dimension(3);

            if (enc_dim != dec_dim || enc_h != dec_h || enc_w != dec_w) {
                std::ostringstream error_message;
                error_message << "Unsupported kernel shape: Encoder shape(" << enc_dim << "x" << enc_channels << "x" << enc_h <<"x" << enc_w <<")\
                                    and Decoder shape(" << dec_dim << "x" << dec_channels << "x" << dec_h <<"x" << dec_w <<")"<<std::endl;
                throw std::runtime_error(error_message.str());
            }

            Tensor4D concatTensor(dec_dim, enc_channels+dec_channels, dec_h, dec_w);
            for(int d=0; d<dec_dim;d++){
                for(int c=0;c<enc_channels;c++){
                    concatTensor(d,c) = encoder(d,c);
                }
                for(int c=enc_channels;c<enc_channels+dec_channels;c++){
                    concatTensor(d,c) = encoder(d,c);
                }
            }
            return concatTensor;
        }

};

int main(){
    int N = 1;  // Batch size
    int C = 3;  // Channels
    int H = 256;  // Height
    int W = 256;  // Width

    Tensor4D input(N, C, H, W);
    // Tensor4D input_tensor2(N, C, H, W);

    // Initialize one of the matrices with random values
    input(0, 0) = Eigen::MatrixXf::Random(H, W);

    Conv2D nn = Conv2D();
    Tensor4D s1 = nn.conv2d(input,"inc_double_conv_0");
    s1.relu();
    s1.printShape();
    return 0;
}