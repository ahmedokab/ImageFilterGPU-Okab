#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>  // ADD THIS LINE

// CUDA kernel launch functions
bool launchBlurKernel(const cv::Mat& input, cv::Mat& output, int radius);
bool launchSharpenKernel(const cv::Mat& input, cv::Mat& output, float strength);
bool launchEdgeKernel(const cv::Mat& input, cv::Mat& output);

// Utility functions
void checkCudaError(const char* operation);
void printKernelLaunchInfo(const char* kernelName, dim3 gridSize, dim3 blockSize);

#endif // CUDA_KERNELS_H
