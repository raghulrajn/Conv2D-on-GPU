#include <iostream>
#include <vector>
#include <cmath> // For sqrt

// Function to compute the mean for each channel
void compute_mean(const std::vector<float>& tensor, std::vector<float>& mean, int N, int C, int H, int W) {
    int spatial_size = H * W;
    int channel_size = N * spatial_size;

    // Initialize mean values to 0
    std::fill(mean.begin(), mean.end(), 0.0f);

    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int hw = 0; hw < spatial_size; ++hw) {
                int idx = n * C * spatial_size + c * spatial_size + hw;
                sum += tensor[idx];
            }
        }
        mean[c] = sum / static_cast<float>(channel_size);
    }
}

// Function to compute the variance for each channel
void compute_variance(const std::vector<float>& tensor, const std::vector<float>& mean, std::vector<float>& variance, int N, int C, int H, int W) {
    int spatial_size = H * W;
    int channel_size = N * spatial_size;

    // Initialize variance values to 0
    std::fill(variance.begin(), variance.end(), 0.0f);

    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        float mean_c = mean[c];
        for (int n = 0; n < N; ++n) {
            for (int hw = 0; hw < spatial_size; ++hw) {
                int idx = n * C * spatial_size + c * spatial_size + hw;
                float diff = tensor[idx] - mean_c;
                sum += diff * diff;
            }
        }
        variance[c] = sum / static_cast<float>(channel_size);
    }
}

// Function to compute the variance for each channel
void compute_batch_norm(std::vector<float>& tensor, const std::vector<float>& mean, std::vector<float>& variance, int N, int C, int H, int W) {
    int spatial_size = H * W;
    int channel_size = N * spatial_size;

   int idx = 0; // Global thread ID
    int total_size = N * C * H * W;

    while (idx < total_size) {
        int c = (idx / (H * W)) % C; // Channel ID
        float mean_c = mean[c];
        float variance_c = variance[c];

        float epsilon = 1e-5f; // Small constant
        tensor[idx] = (tensor[idx] - mean_c) / sqrt(variance_c + epsilon);
        idx++;
    }
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
    // Example tensor dimensions
    int N = 1; // Number of batches
    int C = 3; // Number of channels
    int H = 3; // Height
    int W = 3; // Width

    // Example flattened 4D tensor with random values
    std::vector<float> tensor = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, // Example data for demonstration
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        1, 2, 3, 4, 5, 6, 7, 8, 9

    };

    // Ensure tensor size matches the given dimensions
    int total_elements = N * C * H * W;
    if (tensor.size() != total_elements) {
        std::cerr << "Error: Tensor size does not match the specified dimensions." << std::endl;
        return -1;
    }

    // Vectors to store the results
    std::vector<float> mean(C, 0.0f);      // One mean per channel
    std::vector<float> variance(C, 0.0f); // One variance per channel

    // Compute mean and variance
    compute_mean(tensor, mean, N, C, H, W);
    compute_variance(tensor, mean, variance, N, C, H, W);
    compute_batch_norm(tensor, mean, variance, N, C, H, W);

    // Print the results
    std::cout << "Mean for each channel:" << std::endl;
    for (int c = 0; c < C; ++c) {
        std::cout << "Channel " << c << ": " << mean[c] << std::endl;
    }

    std::cout << "Variance for each channel:" << std::endl;
    for (int c = 0; c < C; ++c) {
        std::cout << "Channel " << c << ": " << variance[c] << std::endl;
    }

    printFlattened4DTensor(tensor,N,C,H,W);
    return 0;
}
