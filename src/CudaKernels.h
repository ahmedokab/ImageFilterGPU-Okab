#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

/*
 * Max blur radius supported by the tiled kernel.
 * Shared memory per block = (BLOCK_W + 2*MAX_RADIUS) * (BLOCK_H + 2*MAX_RADIUS) * 4 bytes.
 * At MAX_RADIUS=16, BLOCK=16:  48 * 48 * 4 = 9,216 bytes — well within the 48 KB limit.
 * Increase if you need larger radii (check shared memory usage with --ptxas-options=-v).
 */
#define MAX_RADIUS 16

// CUDA kernel launch functions — drop-in replacements for the naive versions
bool launchBlurKernel      (const cv::Mat& input, cv::Mat& output, int radius);
bool launchSharpenKernel   (const cv::Mat& input, cv::Mat& output, float strength);
bool launchEdgeKernel      (const cv::Mat& input, cv::Mat& output);
bool launchGrayscaleKernel (const cv::Mat& input, cv::Mat& output);

// Utility functions
void checkCudaError       (const char* operation);
void printKernelLaunchInfo(const char* kernelName, dim3 gridSize, dim3 blockSize);

#endif // CUDA_KERNELS_H
