#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "❌ CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

// TODO: IMPLEMENT THIS BLUR KERNEL
__global__ void blurKernel(unsigned char* input, unsigned char* output,
                          int width, int height, int channels, int radius) {
    // YOUR IMPLEMENTATION HERE
    // Hints:
    // 1. Calculate thread position: int x = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. Check bounds: if (x >= width || y >= height) return;
    // 3. Apply blur averaging in radius around each pixel
    // 4. Handle image boundaries (clamp, wrap, or mirror)
}

// TODO: IMPLEMENT THIS SHARPEN KERNEL  
__global__ void sharpenKernel(unsigned char* input, unsigned char* output,
                             int width, int height, int channels, float strength) {
    // YOUR IMPLEMENTATION HERE
    // Hints:
    // 1. Use convolution with sharpening kernel
    // 2. Center weight: (4 * strength + 1), Neighbors: -strength
    // 3. Clamp results to [0, 255] range
}

// TODO: IMPLEMENT THIS EDGE DETECTION KERNEL
__global__ void edgeKernel(unsigned char* input, unsigned char* output,
                          int width, int height, int channels) {
    // YOUR IMPLEMENTATION HERE  
    // Hints:
    // 1. Implement Sobel operator (or other edge detection)
    // 2. Sobel X: {-1,0,1,-2,0,2,-1,0,1}
    // 3. Sobel Y: {-1,-2,-1,0,0,0,1,2,1}
    // 4. Magnitude = sqrt(gx^2 + gy^2)
}

// Kernel launch functions - IMPLEMENT THESE
bool launchBlurKernel(const cv::Mat& input, cv::Mat& output, int radius) {
    // TODO: YOUR GPU MEMORY MANAGEMENT AND KERNEL LAUNCH
    // 1. Create output Mat
    // 2. Allocate GPU memory
    // 3. Copy input to GPU
    // 4. Launch kernel with proper grid/block dimensions
    // 5. Copy result back to CPU
    // 6. Free GPU memory
    
    // For now, return false so CPU fallback is used
    std::cout << "⚠️  Blur kernel not implemented - using CPU fallback" << std::endl;
    return false;
}

bool launchSharpenKernel(const cv::Mat& input, cv::Mat& output, float strength) {
    // TODO: IMPLEMENT SHARPEN KERNEL LAUNCH
    std::cout << "⚠️  Sharpen kernel not implemented - using CPU fallback" << std::endl;
    return false;
}

bool launchEdgeKernel(const cv::Mat& input, cv::Mat& output) {
    // TODO: IMPLEMENT EDGE KERNEL LAUNCH
    std::cout << "⚠️  Edge kernel not implemented - using CPU fallback" << std::endl;
    return false;
}

// Utility functions (already implemented for you)
void checkCudaError(const char* operation) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "❌ CUDA error in " << operation << ": " 
                  << cudaGetErrorString(error) << std::endl;
    }
}

void printKernelLaunchInfo(const char* kernelName, dim3 gridSize, dim3 blockSize) {
    std::cout << "🚀 Launching " << kernelName << " kernel:" << std::endl;
    std::cout << "   Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "   Block size: " << blockSize.x << "x" << blockSize.y << std::endl;
    std::cout << "   Total threads: " << gridSize.x * gridSize.y * blockSize.x * blockSize.y << std::endl;
}