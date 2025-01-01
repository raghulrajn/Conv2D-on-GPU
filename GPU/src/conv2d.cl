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