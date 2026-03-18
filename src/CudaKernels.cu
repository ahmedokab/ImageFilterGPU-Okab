/*
 * CudaKernels.cu -- ImageFilterGPU-Okab
 * Ahmed Okab
 *
 * Two sets of launch functions:
 *
 *   Standard (launchBlurKernel etc.)
 *     Handle cudaMalloc, cudaMemcpy H->D, kernel, cudaMemcpy D->H, cudaFree.
 *     Used by ImageProcessor for single-filter calls.
 *
 *   Raw (launchBlurKernelRaw etc.)
 *     Accept device pointers directly. No allocation, no transfer.
 *     Used by GpuImage for pipeline mode -- data stays on GPU between calls.
 *
 * Kernel optimizations:
 *   blur      -- shared memory tiling (87x fewer global reads at radius=10)
 *   sharpen   -- __ldg read-only cache + #pragma unroll on 3x3
 *   edge      -- __ldg read-only cache + #pragma unroll on 3x3
 *   grayscale -- __ldg + fully unrolled (no channel loop)
 */

#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -- " << cudaGetErrorString(_e) << std::endl;        \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define BLOCK_W 16
#define BLOCK_H 16

// ============================================================================
// KERNELS
// ============================================================================

// Blur: shared memory tiling
// Each block loads (BLOCK_W + 2r) x (BLOCK_H + 2r) into __shared__ once.
// All neighbor reads come from on-chip shared memory (~1 cycle vs ~200 cycles).
__global__ void blurKernelTiled(const unsigned char* __restrict__ input,
                                unsigned char* __restrict__ output,
                                int width, int height, int channels, int radius)
{
    extern __shared__ float smem[];

    const int tileW = blockDim.x + 2 * radius;
    const int tileH = blockDim.y + 2 * radius;
    const int x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int lx    = threadIdx.x + radius;
    const int ly    = threadIdx.y + radius;

    for (int c = 0; c < channels; c++) {
        for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y)
            for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
                int gx = max(0, min(blockIdx.x * blockDim.x + tx - radius, width  - 1));
                int gy = max(0, min(blockIdx.y * blockDim.y + ty - radius, height - 1));
                smem[ty * tileW + tx] = __ldg(&input[(gy * width + gx) * channels + c]);
            }
        __syncthreads();

        if (x < width && y < height) {
            float sum = 0.0f;
            const int count = (2 * radius + 1) * (2 * radius + 1);
            for (int yp = -radius; yp <= radius; yp++)
                for (int xp = -radius; xp <= radius; xp++)
                    sum += smem[(ly + yp) * tileW + (lx + xp)];
            output[(y * width + x) * channels + c] = (unsigned char)(sum / count);
        }
        __syncthreads();
    }
}

// Sharpen: __ldg + unrolled 3x3 convolution
__global__ void sharpenKernelOptimized(const unsigned char* __restrict__ input,
                                       unsigned char* __restrict__ output,
                                       int width, int height, int channels,
                                       float strength)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        float ans = 0.0f;
        #pragma unroll
        for (int yp = -1; yp <= 1; yp++) {
            #pragma unroll
            for (int xp = -1; xp <= 1; xp++) {
                int nx = max(0, min(x + xp, width  - 1));
                int ny = max(0, min(y + yp, height - 1));
                float pv = __ldg(&input[(ny * width + nx) * channels + c]);
                ans += (xp == 0 && yp == 0) ? pv * (4.0f * strength + 1.0f)
                                             : pv * (-strength * 0.5f);
            }
        }
        output[(y * width + x) * channels + c] =
            (unsigned char)fmaxf(0.0f, fminf(255.0f, ans));
    }
}

// Edge: Sobel with __ldg + unrolled 3x3
__global__ void edgeKernelOptimized(const unsigned char* __restrict__ input,
                                    unsigned char* __restrict__ output,
                                    int width, int height, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int sobelY[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

    for (int c = 0; c < channels; c++) {
        float gx = 0.0f, gy = 0.0f;
        #pragma unroll
        for (int yp = -1; yp <= 1; yp++) {
            #pragma unroll
            for (int xp = -1; xp <= 1; xp++) {
                int nx = max(0, min(x + xp, width  - 1));
                int ny = max(0, min(y + yp, height - 1));
                float pv = __ldg(&input[(ny * width + nx) * channels + c]);
                gx += pv * sobelX[yp + 1][xp + 1];
                gy += pv * sobelY[yp + 1][xp + 1];
            }
        }
        output[(y * width + x) * channels + c] =
            (unsigned char)fminf(255.0f, sqrtf(gx * gx + gy * gy));
    }
}

// Grayscale: fully unrolled, no channel loop
__global__ void grayscaleKernelOptimized(const unsigned char* __restrict__ input,
                                         unsigned char* __restrict__ output,
                                         int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int base = (y * width + x) * 3;
    float r = __ldg(&input[base + 0]);
    float g = __ldg(&input[base + 1]);
    float b = __ldg(&input[base + 2]);
    unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    output[base + 0] = gray;
    output[base + 1] = gray;
    output[base + 2] = gray;
}

// ============================================================================
// SHARED HELPERS
// ============================================================================

static bool allocAndTransfer(const cv::Mat& input,
                              unsigned char** d_in, unsigned char** d_out,
                              size_t& imgSz)
{
    imgSz = (size_t)input.cols * input.rows * input.channels();
    if (cudaMalloc(d_in,  imgSz) != cudaSuccess) return false;
    if (cudaMalloc(d_out, imgSz) != cudaSuccess) { cudaFree(*d_in); return false; }
    if (cudaMemcpy(*d_in, input.data, imgSz,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*d_in); cudaFree(*d_out); return false;
    }
    return true;
}

static bool retrieveAndFree(cv::Mat& output,
                             unsigned char* d_in, unsigned char* d_out,
                             size_t imgSz)
{
    bool ok = (cudaDeviceSynchronize() == cudaSuccess) &&
              (cudaMemcpy(output.data, d_out, imgSz,
                          cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d_in);
    cudaFree(d_out);
    return ok;
}

static void makeDims(int W, int H, dim3& block, dim3& grid)
{
    block = dim3(BLOCK_W, BLOCK_H);
    grid  = dim3((W + BLOCK_W - 1) / BLOCK_W, (H + BLOCK_H - 1) / BLOCK_H);
}

// ============================================================================
// RAW LAUNCH FUNCTIONS (device pointers in, device pointers out -- no transfer)
// Called by GpuImage for pipeline mode.
// ============================================================================

bool launchBlurKernelRaw(unsigned char* d_in, unsigned char* d_out,
                          int width, int height, int channels, int radius)
{
    if (radius > MAX_RADIUS) {
        std::cerr << "Radius " << radius << " > MAX_RADIUS=" << MAX_RADIUS << "\n";
        return false;
    }
    dim3 block, grid;
    makeDims(width, height, block, grid);
    const int tileW = BLOCK_W + 2 * radius;
    const int tileH = BLOCK_H + 2 * radius;
    size_t smemSz   = (size_t)tileW * tileH * sizeof(float);
    blurKernelTiled<<<grid, block, smemSz>>>(d_in, d_out, width, height, channels, radius);
    return cudaGetLastError() == cudaSuccess;
}

bool launchSharpenKernelRaw(unsigned char* d_in, unsigned char* d_out,
                             int width, int height, int channels, float strength)
{
    dim3 block, grid;
    makeDims(width, height, block, grid);
    sharpenKernelOptimized<<<grid, block>>>(d_in, d_out, width, height, channels, strength);
    return cudaGetLastError() == cudaSuccess;
}

bool launchEdgeKernelRaw(unsigned char* d_in, unsigned char* d_out,
                          int width, int height, int channels)
{
    dim3 block, grid;
    makeDims(width, height, block, grid);
    edgeKernelOptimized<<<grid, block>>>(d_in, d_out, width, height, channels);
    return cudaGetLastError() == cudaSuccess;
}

bool launchGrayscaleKernelRaw(unsigned char* d_in, unsigned char* d_out,
                               int width, int height)
{
    dim3 block, grid;
    makeDims(width, height, block, grid);
    grayscaleKernelOptimized<<<grid, block>>>(d_in, d_out, width, height);
    return cudaGetLastError() == cudaSuccess;
}

// ============================================================================
// STANDARD LAUNCH FUNCTIONS (include full host<->device transfer)
// Called by ImageProcessor for single-filter mode.
// ============================================================================

bool launchBlurKernel(const cv::Mat& input, cv::Mat& output, int radius)
{
    output = cv::Mat::zeros(input.size(), input.type());
    if (radius > MAX_RADIUS) {
        std::cerr << "Radius " << radius << " > MAX_RADIUS=" << MAX_RADIUS << "\n";
        return false;
    }
    unsigned char *d_in = nullptr, *d_out = nullptr;
    size_t imgSz;
    if (!allocAndTransfer(input, &d_in, &d_out, imgSz)) return false;
    dim3 block, grid;
    makeDims(input.cols, input.rows, block, grid);
    const int tileW = BLOCK_W + 2 * radius;
    const int tileH = BLOCK_H + 2 * radius;
    size_t smemSz   = (size_t)tileW * tileH * sizeof(float);
    printKernelLaunchInfo("Blur (tiled)", grid, block);
    blurKernelTiled<<<grid, block, smemSz>>>(d_in, d_out,
                                              input.cols, input.rows,
                                              input.channels(), radius);
    return retrieveAndFree(output, d_in, d_out, imgSz);
}

bool launchSharpenKernel(const cv::Mat& input, cv::Mat& output, float strength)
{
    output = cv::Mat::zeros(input.size(), input.type());
    unsigned char *d_in = nullptr, *d_out = nullptr;
    size_t imgSz;
    if (!allocAndTransfer(input, &d_in, &d_out, imgSz)) return false;
    dim3 block, grid;
    makeDims(input.cols, input.rows, block, grid);
    printKernelLaunchInfo("Sharpen (__ldg+unroll)", grid, block);
    sharpenKernelOptimized<<<grid, block>>>(d_in, d_out,
                                             input.cols, input.rows,
                                             input.channels(), strength);
    return retrieveAndFree(output, d_in, d_out, imgSz);
}

bool launchEdgeKernel(const cv::Mat& input, cv::Mat& output)
{
    output = cv::Mat::zeros(input.size(), input.type());
    unsigned char *d_in = nullptr, *d_out = nullptr;
    size_t imgSz;
    if (!allocAndTransfer(input, &d_in, &d_out, imgSz)) return false;
    dim3 block, grid;
    makeDims(input.cols, input.rows, block, grid);
    printKernelLaunchInfo("Edge (__ldg+unroll)", grid, block);
    edgeKernelOptimized<<<grid, block>>>(d_in, d_out,
                                          input.cols, input.rows,
                                          input.channels());
    return retrieveAndFree(output, d_in, d_out, imgSz);
}

bool launchGrayscaleKernel(const cv::Mat& input, cv::Mat& output)
{
    output = cv::Mat::zeros(input.size(), input.type());
    unsigned char *d_in = nullptr, *d_out = nullptr;
    size_t imgSz;
    if (!allocAndTransfer(input, &d_in, &d_out, imgSz)) return false;
    dim3 block, grid;
    makeDims(input.cols, input.rows, block, grid);
    printKernelLaunchInfo("Grayscale (unrolled)", grid, block);
    grayscaleKernelOptimized<<<grid, block>>>(d_in, d_out,
                                               input.cols, input.rows);
    return retrieveAndFree(output, d_in, d_out, imgSz);
}

// ============================================================================
// UTILITY
// ============================================================================

void checkCudaError(const char* operation)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        std::cerr << "CUDA error in " << operation << ": "
                  << cudaGetErrorString(error) << std::endl;
}

void printKernelLaunchInfo(const char* kernelName, dim3 gridSize, dim3 blockSize)
{
    std::cout << "Launching " << kernelName << " kernel:\n"
              << "   Grid:    " << gridSize.x  << " x " << gridSize.y  << "\n"
              << "   Block:   " << blockSize.x << " x " << blockSize.y << "\n"
              << "   Threads: "
              << (long long)gridSize.x * gridSize.y * blockSize.x * blockSize.y
              << "\n";
}
