#include <Core/Assert.hpp>
#include <Core/Image.hpp>
#include <Core/Time.hpp>
#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Error.hpp>
#include <OpenCL/cl-patched.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

class GPUInit {

	private:
		cl::Context context;
		cl::CommandQueue queue;
		cl::Program program;
		cl::Device device;
		std::vector<cl::Device> devices;

		cl::Kernel convKernel;
		cl::Kernel meanKernel;
		cl::Kernel varianceKernel;
		cl::Kernel reluKernel;
		cl::Kernel batchNormKernel;

		// Timing events
		cl::Event copyToDeviceEvent;
		cl::Event kernelEvent;
		cl::Event copyFromDeviceEvent;

		std::string load_kernel_source(const std::string& filename) {
			std::ifstream file(filename);
			if (!file.is_open()) {
				std::cerr << "Failed to open kernel source file." << std::endl;
				exit(1);
			}
			return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
		}

	public:
	GPUInit(){
		context = cl::Context(CL_DEVICE_TYPE_GPU);
		device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		devices.push_back(device);
		
		// Create a command queue
		queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

		// Load the source code
		extern unsigned char conv2d_cl[];
		extern unsigned int conv2d_cl_len;
		// std::string kernel_source = load_kernel_source("/home/raghul/Desktop/Conv2D-on-GPU/GPU/src/conv2d.cl");
		// cl::Program program(context, kernel_source);
		cl::Program program(context, std::string((const char*)conv2d_cl, conv2d_cl_len));
		// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
		OpenCL::buildProgram(program, devices);
		std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size()<< " devices" << std::endl;
		OpenCL::printDeviceInfo(std::cout, device);
		// cl::Kernel meanKernel(program, "compute_mean");
		try {
			meanKernel = cl::Kernel(program, "compute_mean");
			varianceKernel = cl::Kernel(program, "compute_variance");
			reluKernel = cl::Kernel(program, "compute_relu");
			batchNormKernel = cl::Kernel(program, "compute_batch_norm");
		} catch (OpenCL::Error &e) {
			std::cerr << "Error creating kernel: " << e.what() << std::endl;
			throw std::runtime_error("Failed to create kernel.");
		}

	}

	void convolve4D(const float* input,const float* kernel,float* output,int batchSize,int inChannels,int outChannels,int inHeight,int inWidth,
        int kernelH,
        int kernelW,
        int stride = 1,
        int padding = 0) {
        	auto startTotal = std::chrono::high_resolution_clock::now();
			cl::Kernel convKernel(program, "conv2d");
            // Calculate output dimensions
            int outHeight = (inHeight + 2 * padding - kernelH) / stride + 1;
            int outWidth = (inWidth + 2 * padding - kernelW) / stride + 1;
            
            // Create buffers
            size_t inputSize = batchSize * inChannels * inHeight * inWidth * sizeof(float);
            size_t kernelSize = outChannels * inChannels * kernelH * kernelW * sizeof(float);
            size_t outputSize = batchSize * outChannels * outHeight * outWidth * sizeof(float);
            
            cl::Buffer d_input(context, CL_MEM_READ_ONLY, inputSize);
            cl::Buffer d_kernel(context, CL_MEM_READ_ONLY, kernelSize);
            cl::Buffer d_output(context, CL_MEM_WRITE_ONLY, outputSize);
            
            // Copy data to device
            auto startCopy = std::chrono::high_resolution_clock::now();
            queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, inputSize, input, 
                                   nullptr, NULL);
            queue.enqueueWriteBuffer(d_kernel, CL_TRUE, 0, kernelSize, kernel);
            auto endCopy = std::chrono::high_resolution_clock::now();
            
            // Set kernel arguments
            convKernel.setArg(0, d_input);
            convKernel.setArg(1, d_kernel);
            convKernel.setArg(2, d_output);
            convKernel.setArg(3, batchSize);
            convKernel.setArg(4, inChannels);
            convKernel.setArg(5, outChannels);
            convKernel.setArg(6, inHeight);
            convKernel.setArg(7, inWidth);
            convKernel.setArg(8, kernelH);
            convKernel.setArg(9, kernelW);
            convKernel.setArg(10, outHeight);
            convKernel.setArg(11, outWidth);
            convKernel.setArg(12, stride);
            convKernel.setArg(13, padding);
            
            // Execute kernel
            auto startCompute = std::chrono::high_resolution_clock::now();
            
            // Calculate work group size
            cl::NDRange globalSize(outWidth, outHeight, outChannels);
            cl::NDRange localSize(16, 16, 1); // Adjust based on your GPU
            
            queue.enqueueNDRangeKernel(convKernel, cl::NullRange, globalSize, localSize, nullptr, NULL);
            auto endCompute = std::chrono::high_resolution_clock::now();
            // Read back result
            auto startReadBack = std::chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_output, CL_TRUE, 0, outputSize, output,nullptr, NULL);
            auto endReadBack = std::chrono::high_resolution_clock::now();
            
            } 

	std::vector<float> computeMean(std::vector<float> input, std::vector<float> mean, int N, int C, int H, int W){

		cl::NDRange globalSize(N, H, W);    // Global size
		// cl::NDRange localSize(N / C, 1, 1); // Adjust local size for reduction

		// Allocate local memory (size matches localSize[0])
		size_t localMemSize = sizeof(float) * 256;

		cl::Buffer tensor_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
        cl::Buffer mean_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * mean.size(), mean.data());

		// Set kernel arguments		
		meanKernel.setArg(0, tensor_buffer);
		meanKernel.setArg(1, mean_buffer);
		// meanKernel.setArg(2, cl::Local(localMemSize)); // Local memory
		meanKernel.setArg(2, N);
		meanKernel.setArg(3, C);
		meanKernel.setArg(4, H);
		meanKernel.setArg(5, W);

		// Launch the kernel
		queue.enqueueNDRangeKernel(meanKernel,cl::NDRange(),cl::NDRange(C,1,1),cl::NDRange());

		queue.enqueueReadBuffer(mean_buffer, CL_TRUE, 0, sizeof(float) * mean.size(), mean.data());

		 std::cout << "Mean for each channel:" << std::endl;
        for (int c = 0; c < C; ++c) {
            std::cout << "Channel " << c << ": " << mean[c] << std::endl;
        }
		return mean;
	}

	std::vector<float> computeVariance(std::vector<float> input, std::vector<float> mean,std::vector<float> variance, int N, int C, int H, int W){
		std::vector<float> _mean = computeMean(input, mean, N,C,H,W);
		cl::NDRange globalSize(N, H, W);    // Global size
		// cl::NDRange localSize(N / C, 1, 1); // Adjust local size for reduction

		// Allocate local memory (size matches localSize[0])
		size_t localMemSize = sizeof(float) * 256;

		cl::Buffer tensor_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
        cl::Buffer mean_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * _mean.size(), _mean.data());
		cl::Buffer variance_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * variance.size(), variance.data());
		// Set kernel arguments		
		varianceKernel.setArg(0, tensor_buffer);
		varianceKernel.setArg(1, mean_buffer);
		varianceKernel.setArg(2, variance_buffer);
		varianceKernel.setArg(3, N);
		varianceKernel.setArg(4, C);
		varianceKernel.setArg(5, H);
		varianceKernel.setArg(6, W);

		// Launch the kernel
		queue.enqueueNDRangeKernel(varianceKernel,cl::NDRange(),cl::NDRange(C),cl::NDRange());

		queue.enqueueReadBuffer(variance_buffer, CL_TRUE, 0, sizeof(float) * variance.size(), variance.data());

		 std::cout << "Variance for each channel:" << std::endl;
        for (int c = 0; c < C; ++c) {
            std::cout << "Channel " << c << ": " << variance[c] << std::endl;
        }
		return variance;
	}
	
	void computeRelu(std::vector<float> &input, int N, int C, int H, int W){
		
		cl::NDRange globalSize(N*C*H*W);    // Global size
		
		cl::Buffer tensor_buffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
       
		reluKernel.setArg(0, tensor_buffer);
		reluKernel.setArg(1, N*C*H*W);
		
		// Launch the kernel
		queue.enqueueNDRangeKernel(reluKernel,cl::NDRange(),globalSize,cl::NDRange());

		queue.enqueueReadBuffer(tensor_buffer, CL_TRUE, 0, sizeof(float) * input.size(), input.data());
	}

	std::vector<float> computebatchNorm(std::vector<float> &input, int N, int C, int H, int W){

		cl::NDRange globalSize(N, H, W);    // Global size
		// cl::NDRange localSize(N, H, W); 

		cl::Buffer tensor_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
        cl::Buffer mean_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * C);
		cl::Buffer variance_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * C);

		// Set kernel arguments		
		meanKernel.setArg(0, tensor_buffer);
		meanKernel.setArg(1, mean_buffer);
		meanKernel.setArg(2, N);
		meanKernel.setArg(3, C);
		meanKernel.setArg(4, H);
		meanKernel.setArg(5, W);

		varianceKernel.setArg(0, tensor_buffer);
		varianceKernel.setArg(1, mean_buffer);
		varianceKernel.setArg(2, variance_buffer);
		varianceKernel.setArg(3, N);
		varianceKernel.setArg(4, C);
		varianceKernel.setArg(5, H);
		varianceKernel.setArg(6, W);

		batchNormKernel.setArg(0, tensor_buffer);
        batchNormKernel.setArg(1, mean_buffer);
        batchNormKernel.setArg(2, variance_buffer);
        batchNormKernel.setArg(3, N);
        batchNormKernel.setArg(4, C);
        batchNormKernel.setArg(5, H);
        batchNormKernel.setArg(6, W);

		// Launch the kernel
		queue.enqueueNDRangeKernel(meanKernel,cl::NDRange(),cl::NDRange(C,1,1),cl::NDRange());
		queue.enqueueNDRangeKernel(varianceKernel,cl::NDRange(),cl::NDRange(C,1,1),cl::NDRange());
		queue.enqueueNDRangeKernel(batchNormKernel,cl::NDRange(),cl::NDRange(N*C*H*W),cl::NDRange());

		queue.enqueueReadBuffer(tensor_buffer, CL_TRUE, 0, sizeof(float) * input.size(), input.data());

		return input;
	}

};



// Read image 
static std::vector<float> loadImageToTensor4D(const std::string& imagePath, int targetWidth = -1, int targetHeight = -1, bool normalize = true) {

        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + imagePath);
        }

        // Resize if dimensions are specified
        if (targetWidth > 0 && targetHeight > 0) {
            cv::resize(image, image, cv::Size(targetWidth, targetHeight));
        }

        // Get dimensions
        int batch = 1;
        int channels = 3;  // RGB
        int height = image.rows;
        int width = image.cols;

        // Create 4D vector with NCHW layout
        std::vector<float> tensor(batch * channels * height * width);

        // Convert to float and normalize if requested
        cv::Mat float_img;
        if (normalize) {
            image.convertTo(float_img, CV_32F, 1.0/255.0);
        } else {
            image.convertTo(float_img, CV_32F);
        }

        // Reorder from HWC (OpenCV) to NCHW (tensor format)
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                cv::Vec3f pixel = float_img.at<cv::Vec3f>(h, w);
                for (int c = 0; c < channels; ++c) {
                    // Calculate index in NCHW format
                    size_t tensor_idx = ((0 * channels + c) * height + h) * width + w;
                    tensor[tensor_idx] = pixel[c];
                }
            }
        }

        return tensor;
    }

static std::vector<float> loadKernelToTensor4D(){
	int inChannels = 3;  // RGB image
	int outChannels = 64;
	int kernelH = 3;
	int kernelW = 3;
	std::vector<float> tensor(outChannels * inChannels * kernelH * kernelW);
	std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f); // Random values between -1 and 1

	int total_elements = outChannels * inChannels * kernelH * kernelW;
    // Assign random values to the tensor
    for (int i = 0; i < total_elements; ++i) {
        tensor[i] = distribution(generator);
    }
	return tensor;
}

void printFlattened4DTensor(const std::vector<float>& tensor, int N, int C, int H, int W) {
    // The 4D tensor is flattened as a 1D array of size N*C*H*W
    // To print as a matrix, we can print each slice of the tensor in a matrix format.

    // Loop over the batches (N)
    for (int n = 0; n < N; ++n) {
        std::cout << "Batch " << n << ":\n";
        
        // Loop over the channels (C)
        for (int c = 0; c < C; ++c) {
            std::cout << " Channel " << c << ":\n";
            
            // Loop over the height (H)
            for (int h = 0; h < H; ++h) {
                // Loop over the width (W) and print the corresponding elements
                for (int w = 0; w < W; ++w) {
                    int idx = n * C * H * W + c * H * W + h * W + w; // Calculate the flattened index
                    std::cout << tensor[idx] << " ";
                }
                std::cout << "\n";  // New line after each row of the matrix (height)
            }
        }
    }
}

int main() {
    GPUInit gpu = GPUInit();
	std::cout<<"GPU Initialized\n";
	int N = 1; // Number of batches
    int C = 3; // Number of channels
    int H = 3; // Height
    int W = 3; // Width

    //flattened 4D tensor with random values
    std::vector<float> tensor = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, // Example data for demonstration
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        1, 2, 3, 4, 5, 6, 7, 8, 9

    };

	std::vector<float>mean(C, 0.0f);
	std::vector<float>variance(C, 0.0f);

	variance = gpu.computeVariance(tensor, mean,variance, N, C, H, W);
	gpu.computeRelu(tensor,N,C,H,W);

	std::vector<float> bn = gpu.computebatchNorm(tensor, N, C,H,W);

	printFlattened4DTensor(tensor, N,C,H,W);
	printFlattened4DTensor(bn, N,C,H,W);

    return 0;
} 
