#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

__global__ void blurKernel(unsigned char* input, unsigned char* output,
                          int width, int height, int channels, int radius) {
    // every CUDA thread will process a single pixel, many pixels being processed simultaneously
    int x = blockIdx.x * blockDim.x + threadIdx.x; //finding the column, finding which pixel the thread handles
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; //sanity check to not process out of bounds pixels

    // 1920x1080 image, launch 2 million threads each handling a singular pixel
    // images in memory stored in a 1d array in row-major order

    // loop through all 3 channels in a RGB
    for(int c = 0; c < channels; c++){
        // inorder to blur, sample the pixels inside a square around the current pixel
        float sum = 0.0f;
        int count = 0;
        
        for(int yp = -radius; yp <= radius; yp++){
            for(int xp = -radius; xp <= radius; xp++){
                int nx = x + xp;
                int ny = y + yp; // find the neighbor pixel we want

                nx = max(0, min(nx, width - 1));
                ny = max(0, min(ny, height - 1));
                //clamping handles boundaries of the image

                int NIDX = (ny * width + nx) * channels + c; // using row major order to access the pixel,
                
                sum += input[NIDX];
                count++;
            }
        }
    
        int outputIDX = (y * width + x) * channels + c;
        output[outputIDX] = (unsigned char) (sum / count); //typecasting into 8 bit value
    }
}



__global__ void sharpenKernel(unsigned char* input, unsigned char* output,
                             int width, int height, int channels, float strength) {
    // every CUDA thread will process a single pixel, many pixels being processed simultaneously
    int x = blockIdx.x * blockDim.x + threadIdx.x; //finding the column, finding which pixel the thread handles
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return; //sanity check to not process out of bounds pixels
    
    // 1920x1080 image, launch 2 million threads each handling a singular pixel
    // images in memory stored in a 1d array in row-major order
    
    // loop through all 3 channels in a RGB
    for(int c = 0; c < channels; c++){
        float ans = 0.0f;  // FIXED: Added semicolon
        
        // FIXED: Added both loops and defined radius
        int radius = 1;  // For 3x3 kernel
        for(int yp = -radius; yp <= radius; yp++){  // FIXED: Added yp loop
            for(int xp = -radius; xp <= radius; xp++){
                int nx = x + xp;
                int ny = y + yp; // find the neighbor pixel we want
                
                nx = max(0, min(nx, width - 1));
                ny = max(0, min(ny, height - 1));
                
                int NIDX = (ny * width + nx) * channels + c; // using row major order to access the pixel
                float pixelval = input[NIDX];  // FIXED: Added type declaration
                
                if(xp == 0 && yp == 0){  // FIXED: Check offset, not absolute position
                    ans += pixelval * (4.0f * strength + 1.0f);  // FIXED: Added .0f
                }
                else{
                    ans += pixelval * (-strength * 0.5f); //creates contrast by decreasing neighbour pixels & increasing centre pixel
                }
                //applying convolutional kernel using cuda, meaning center pixel gets higher weight then outskirt pixels get lowered weight to highlight that pixel
            }
        }
        
        ans = fmaxf(0.0f, fminf(255.0f, ans));  // FIXED: Use ans, not result
        //insures ans is clamped to range [0, 255]
        int outputIdx = (y * width + x) * channels + c;
        output[outputIdx] = (unsigned char)ans;  // FIXED: Use ans, not result
    }
}


__global__ void grayscaleKernel(unsigned char* input, unsigned char* output,
                               int width, int height, int channels) {
    // every CUDA thread processes a single pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // for RGB images only (channels = 3)
    if (channels == 3) {
        int inputIdx = (y * width + x) * channels;
        
        // get RGB values
        float r = input[inputIdx + 0];     // red channel
        float g = input[inputIdx + 1];     // green channel  
        float b = input[inputIdx + 2];     // blue channel
        
        // calculate grayscale using standard luminance formula
        // human eye is more sensitive to green, less to blue
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        
        // clamp to valid range
        gray = fminf(255.0f, fmaxf(0.0f, gray));
        
        // store same gray value in all 3 channels to maintain RGB format
        int outputIdx = (y * width + x) * channels;
        output[outputIdx + 0] = (unsigned char)gray;  // R
        output[outputIdx + 1] = (unsigned char)gray;  // G  
        output[outputIdx + 2] = (unsigned char)gray;  // B
    }
}

bool launchGrayscaleKernel(const cv::Mat& input, cv::Mat& output) {
    output = cv::Mat::zeros(input.size(), input.type());
    
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t imageSize = width * height * channels;
    
    std::cout << "Launching CUDA grayscale: " << width << "x" << height << std::endl;
    
    unsigned char *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    
    CUDA_CHECK(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    printKernelLaunchInfo("Grayscale", gridSize, blockSize);
    
    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    std::cout << "GPU grayscale completed successfully" << std::endl;
    return true;
}
                            
__global__ void edgeKernel(unsigned char* input, unsigned char* output,
                          int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Sobel operator kernels for edge detection
    int sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};  // detects vertical edges
    int sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};  // detects horizontal edges
    
    for (int c = 0; c < channels; c++) {
        float gx = 0.0f;  // gradient in x direction
        float gy = 0.0f;  // gradient in y direction
        
        // apply 3x3 Sobel kernels
        for (int yp = -1; yp <= 1; yp++) {
            for (int xp = -1; xp <= 1; xp++) {
                int nx = x + xp;
                int ny = y + yp;
                
                // clamp to image boundaries
                nx = max(0, min(nx, width - 1));
                ny = max(0, min(ny, height - 1));
                
                int inputIdx = (ny * width + nx) * channels + c;
                float pixelValue = input[inputIdx];
                
                // calculate which position in the 3x3 kernel we at
                int kernelIdx = (yp + 1) * 3 + (xp + 1);
                
                gx += pixelValue * sobelX[kernelIdx];  // horizontal gradient
                gy += pixelValue * sobelY[kernelIdx];  // vertical gradient
            }
        }
        
        // calculate gradient magnitude (edge strength)
        float magnitude = sqrtf(gx * gx + gy * gy);
        
        // clamp to valid pixel range
        magnitude = fminf(255.0f, magnitude);
        
        // store result
        int outputIdx = (y * width + x) * channels + c;
        output[outputIdx] = (unsigned char)magnitude;
    }
}

bool launchBlurKernel(const cv::Mat& input, cv::Mat& output, int radius) {
    output = cv::Mat::zeros(input.size(), input.type());

    int width = input.cols;
    int height = input.rows;    // finding width and height of image
    int channels = input.channels(); 
    size_t imageSize = width * height * channels;  //finding size of the image in bytes, size of unsigned char is 1 byte
    
    std::cout << "Launching CUDA blur: " << width << "x" << height 
              << ", radius=" << radius << std::endl;    

    unsigned char *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));

    CUDA_CHECK(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice)); ///copying the input image from the CPU to the GPU

    dim3 blockSize(16, 16);  
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    printKernelLaunchInfo("Blur", gridSize, blockSize);
    
    blurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels, radius);
    
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for kernel to complete
    
    // Copy result from GPU back to CPU
    CUDA_CHECK(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    std::cout << "GPU blur completed successfully" << std::endl;
    return true;
}

bool launchSharpenKernel(const cv::Mat& input, cv::Mat& output, float strength) {
    output = cv::Mat::zeros(input.size(), input.type());

    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t imageSize = width * height * channels;

    std::cout << "Launching CUDA sharpen: " << width << "x" << height 
              << ", strength=" << strength << std::endl;

    unsigned char *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));

    CUDA_CHECK(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);  // You can use your custom 24x24 if you prefer
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    printKernelLaunchInfo("Sharpen", gridSize, blockSize);

    sharpenKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels, strength);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "GPU sharpen completed successfully" << std::endl;
    return true;
}

bool launchEdgeKernel(const cv::Mat& input, cv::Mat& output) {
    output = cv::Mat::zeros(input.size(), input.type());

    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t imageSize = width * height * channels;

    std::cout << "Launching CUDA edge detection: " << width << "x" << height << std::endl;

    unsigned char *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));

    CUDA_CHECK(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    printKernelLaunchInfo("Edge Detection", gridSize, blockSize);

    edgeKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "GPU edge detection completed successfully" << std::endl;
    return true;
}
void checkCudaError(const char* operation) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " 
                  << cudaGetErrorString(error) << std::endl;
    }
}

void printKernelLaunchInfo(const char* kernelName, dim3 gridSize, dim3 blockSize) {
    std::cout << "Launching " << kernelName << " kernel:" << std::endl;
    std::cout << "   Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "   Block size: " << blockSize.x << "x" << blockSize.y << std::endl;
    std::cout << "   Total threads: " << gridSize.x * gridSize.y * blockSize.x * blockSize.y << std::endl;
}