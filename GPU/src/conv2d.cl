// tensor_kernels.cl
// Helper function to get 4D tensor index
inline int get_tensor_idx(int n, int c, int h, int w,
                         int channels, int height, int width) {
    return ((n * channels + c) * height + h) * width + w;
}

// Main convolution kernel for 4D tensors
__kernel void conv2d(
    __global const float* input,      // Input tensor (N x C_in x H x W)
    __global const float* ker,     // Kernel tensor (C_out x C_in x kH x kW)
    __global float* output,          // Output tensor
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_h,
    const int kernel_w,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding
) {
    // Get global position
    const int ow = get_global_id(0);  // Width position
    const int oh = get_global_id(1);  // Height position
    const int oc = get_global_id(2);  // Output channel
    
    // Early return if out of bounds
    if (ow >= out_width || oh >= out_height || oc >= out_channels)
        return;
    
    // Local memory for kernel weights
    __local float local_kernel[16][16];  // Adjust size based on your needs
    
    // Process each item in batch
    for (int n = 0; n < batch_size; n++) {
        float sum = 0.0f;
        
        // Convolution operation
        for (int ic = 0; ic < in_channels; ic++) {
            // Load kernel weights to local memory
            for (int kh = 0; kh < kernel_h && kh < 16; kh++) {
                for (int kw = 0; kw < kernel_w && kw < 16; kw++) {
                    local_kernel[kh][kw] = ker[get_tensor_idx(oc, ic, kh, kw,
                                                               in_channels, kernel_h, kernel_w)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Perform convolution
            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride + kh - padding;
                if (ih >= 0 && ih < in_height) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int iw = ow * stride + kw - padding;
                        if (iw >= 0 && iw < in_width) {
                            float input_val = input[get_tensor_idx(n, ic, ih, iw,
                                                                 in_channels, in_height, in_width)];
                            sum += input_val * local_kernel[kh][kw];
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Store result
        output[get_tensor_idx(n, oc, oh, ow,
                            out_channels, out_height, out_width)] = sum;
    }
}

__kernel void relu(__global float* tensor, int total_size) {
    int idx = get_global_id(0); // Get global thread ID
    if (idx < total_size) {
        tensor[idx] = fmax(0.0f, tensor[idx]); // Apply ReLU
    }
}


__kernel void compute_mean(
    __global const float* tensor,
    __global float* mean,
    __local float* local_sum, // Shared memory for partial sums
    int N, int C, int H, int W) {

    // Global IDs
    int n = get_global_id(0); // Batch index
    int h = get_global_id(1); // Height index
    int w = get_global_id(2); // Width index

    // Local IDs
    int local_id = get_local_id(0); // Local thread index in workgroup

    // Group ID
    int c = get_group_id(0); // Channel ID (one group per channel)

    // Calculate spatial index
    int spatial_idx = h * W + w;
    int spatial_size = H * W;

    // Flattened index in the tensor
    int idx = n * C * spatial_size + c * spatial_size + spatial_idx;

    // Step 1: Each work-item computes its contribution to the sum
    float partial_sum = 0.0f;

    if (n < N && h < H && w < W) { // Ensure valid bounds
        partial_sum = tensor[idx];
    }

    // Step 2: Write partial sums to local memory
    local_sum[local_id] = partial_sum;

    // Synchronize all work-items in the workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Perform a reduction within the workgroup
    for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            local_sum[local_id] += local_sum[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Step 4: Write the result of the reduction to the global mean array
    if (local_id == 0) { // One work-item per group writes the final sum
        float total_sum = local_sum[0];

        // Normalize by the total number of elements in the channel
        int channel_size = N * spatial_size;
        mean[c] = total_sum / (float)(channel_size);
    }
}

__kernel void compute_variance(
    __global const float* tensor,
    __global const float* mean,
    __global float* variance,
    __local float* local_sum, // Shared memory for partial sums
    int N, int C, int H, int W) {

    // Global IDs
    int n = get_global_id(0); // Batch index
    int h = get_global_id(1); // Height index
    int w = get_global_id(2); // Width index

    // Local ID and group ID
    int local_id = get_local_id(0); // Local thread ID in workgroup
    int c = get_group_id(0);        // Channel ID (one group per channel)

    // Calculate spatial index
    int spatial_idx = h * W + w;
    int spatial_size = H * W;

    // Flattened index in the tensor
    int idx = n * C * spatial_size + c * spatial_size + spatial_idx;

    // Initialize partial sum
    float partial_sum = 0.0f;

    // Read the mean for this channel
    float mean_c = mean[c];

    // Compute partial variance contribution
    if (n < N && h < H && w < W) { // Bounds check
        float diff = tensor[idx] - mean_c;
        partial_sum = diff * diff;
    }

    // Write partial sum to local memory
    local_sum[local_id] = partial_sum;

    // Synchronize all work-items in the workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction within the workgroup
    for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            local_sum[local_id] += local_sum[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the final result for the channel
    if (local_id == 0) {
        float total_sum = local_sum[0];

        // Normalize by the total number of elements in the channel
        int channel_size = N * spatial_size;
        variance[c] = total_sum / (float)(channel_size);
    }
}


