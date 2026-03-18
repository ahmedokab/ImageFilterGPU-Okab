#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

/*
 * Max blur radius for tiled kernel.
 * Shared mem per block = (16 + 2*MAX_RADIUS)^2 * 4 bytes
 * At MAX_RADIUS=16: (48^2)*4 = 9,216 bytes -- well under the 48 KB limit.
 */
#define MAX_RADIUS 16

// ---------------------------------------------------------------------------
// Standard launch functions (handle host<->device transfer internally)
// Used by ImageProcessor for single-filter calls.
// ---------------------------------------------------------------------------
bool launchBlurKernel      (const cv::Mat& input, cv::Mat& output, int radius);
bool launchSharpenKernel   (const cv::Mat& input, cv::Mat& output, float strength);
bool launchEdgeKernel      (const cv::Mat& input, cv::Mat& output);
bool launchGrayscaleKernel (const cv::Mat& input, cv::Mat& output);

// ---------------------------------------------------------------------------
// Raw launch functions (operate directly on device pointers, no transfer)
// Used by GpuImage for pipeline mode. Data stays on GPU between calls.
// ---------------------------------------------------------------------------
bool launchBlurKernelRaw      (unsigned char* d_in, unsigned char* d_out,
                                int width, int height, int channels, int radius);
bool launchSharpenKernelRaw   (unsigned char* d_in, unsigned char* d_out,
                                int width, int height, int channels, float strength);
bool launchEdgeKernelRaw      (unsigned char* d_in, unsigned char* d_out,
                                int width, int height, int channels);
bool launchGrayscaleKernelRaw (unsigned char* d_in, unsigned char* d_out,
                                int width, int height);

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
void checkCudaError       (const char* operation);
void printKernelLaunchInfo(const char* kernelName, dim3 gridSize, dim3 blockSize);

#endif // CUDA_KERNELS_H