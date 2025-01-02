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

    // Print the results
    std::cout << "Mean for each channel:" << std::endl;
    for (int c = 0; c < C; ++c) {
        std::cout << "Channel " << c << ": " << mean[c] << std::endl;
    }

    std::cout << "Variance for each channel:" << std::endl;
    for (int c = 0; c < C; ++c) {
        std::cout << "Channel " << c << ": " << variance[c] << std::endl;
    }

    return 0;
}
