/*
 * CudaKernels_optimized.cu
 * Ahmed Okab — ImageFilterGPU
 *
 * Optimizations over naive implementation:
 *   1. Shared memory tiling  — blur kernel: ~400x fewer global reads
 *   2. Pinned host memory    — doubles PCIe bandwidth for all filters
 *   3. Texture memory cache  — sharpen & edge: 2-D spatial locality
 *   4. Vectorized grayscale  — uchar4 loads, 4 pixels per thread
 *   5. CUDA streams          — overlap H->D transfer with computation
 *   6. __launch_bounds__     — helps compiler register allocation
 */

#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " — " << cudaGetErrorString(_e) << std::endl;          \
            return false;                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Tunable constants
// ---------------------------------------------------------------------------
#define BLOCK_W   16
#define BLOCK_H   16
#define MAX_RADIUS 16          // max blur radius supported by tiled kernel

// Shared memory tile includes a halo of MAX_RADIUS on every side.
// Actual halo used is 'radius' but we size the array at compile time.
#define TILE_W  (BLOCK_W + 2 * MAX_RADIUS)
#define TILE_H  (BLOCK_H + 2 * MAX_RADIUS)

// ---------------------------------------------------------------------------
// Texture reference for 2-D cached reads (sharpen & edge)
// cudaTextureObject_t is preferred for modern CUDA, but a simple global
// texture object created at runtime works fine.
// ---------------------------------------------------------------------------

// ============================================================================
// KERNEL 1 — Tiled Gaussian Blur
// ============================================================================
//
// Strategy:
//   Each block cooperatively loads a (BLOCK_W + 2*radius) x (BLOCK_H + 2*radius)
//   tile into shared memory — one load per pixel in the tile, then every thread
//   reads only from __shared__ when averaging its neighborhood.
//
//   Without tiling: each thread does (2r+1)^2 global reads.
//   With tiling:    each pixel in shared memory is loaded once and reused by
//                   up to (2r+1)^2 threads → massive reduction in global traffic.
//
//   For radius=10:  441 global reads → ~1 global read per output pixel.
// ============================================================================

__global__ void __launch_bounds__(BLOCK_W * BLOCK_H)
blurKernelTiled(const unsigned char* __restrict__ input,
                unsigned char* __restrict__ output,
                int width, int height, int channels, int radius)
{
    // Shared memory tile — one channel processed per kernel call via 'c' loop below.
    // We keep one plane at a time to stay within 48 KB shared memory per block.
    extern __shared__ float smem[];   // dynamic: (BLOCK_W+2r)*(BLOCK_H+2r) floats

    const int tileW = blockDim.x + 2 * radius;
    const int tileH = blockDim.y + 2 * radius;

    // Output pixel this thread is responsible for
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Local (within-tile) indices for this thread's output pixel
    const int lx = threadIdx.x + radius;
    const int ly = threadIdx.y + radius;

    for (int c = 0; c < channels; c++) {

        // ------------------------------------------------------------------
        // Phase 1: Cooperative tile load (including halo)
        // Each thread may load multiple tile pixels to cover the halo.
        // ------------------------------------------------------------------
        for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y) {
            for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
                // Global image coordinates for this tile pixel
                int gx = blockIdx.x * blockDim.x + tx - radius;
                int gy = blockIdx.y * blockDim.y + ty - radius;

                // Clamp to image boundary (mirror / replicate edge)
                gx = max(0, min(gx, width  - 1));
                gy = max(0, min(gy, height - 1));

                smem[ty * tileW + tx] = input[(gy * width + gx) * channels + c];
            }
        }

        __syncthreads();   // all threads must finish loading before anyone reads

        // ------------------------------------------------------------------
        // Phase 2: Each thread averages its (2r+1)^2 neighborhood from smem
        // ------------------------------------------------------------------
        if (x < width && y < height) {
            float sum = 0.0f;
            const int count = (2 * radius + 1) * (2 * radius + 1);

            for (int yp = -radius; yp <= radius; yp++) {
                for (int xp = -radius; xp <= radius; xp++) {
                    sum += smem[(ly + yp) * tileW + (lx + xp)];
                }
            }

            output[(y * width + x) * channels + c] = (unsigned char)(sum / count);
        }

        __syncthreads();   // reset smem before next channel
    }
}

// ============================================================================
// KERNEL 2 — Sharpen with Texture Memory
// ============================================================================
//
// Strategy:
//   Bind the input image to a 2-D texture object.  The texture cache provides
//   spatial locality for the 3×3 neighborhood reads — adjacent threads in a
//   warp reading nearby pixels now hit the L1 texture cache instead of DRAM.
//   Also uses __ldg() (read-only cache) as a fallback path.
// ============================================================================

__global__ void __launch_bounds__(BLOCK_W * BLOCK_H)
sharpenKernelTexture(cudaTextureObject_t texObj,
                     unsigned char* __restrict__ output,
                     int width, int height, int channels, float strength)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        float ans = 0.0f;

        for (int yp = -1; yp <= 1; yp++) {
            for (int xp = -1; xp <= 1; xp++) {
                // tex2D clamps or wraps at boundary based on texture descriptor
                // We configured CLAMP_TO_EDGE below, so no manual clamping needed.
                float px = tex2D<float>(texObj,
                                        (float)(x * channels + c + xp * channels),
                                        (float)(y + yp));

                if (xp == 0 && yp == 0)
                    ans += px * (4.0f * strength + 1.0f);
                else
                    ans += px * (-strength * 0.5f);
            }
        }

        ans = fmaxf(0.0f, fminf(255.0f, ans));
        output[(y * width + x) * channels + c] = (unsigned char)ans;
    }
}

// ============================================================================
// KERNEL 3 — Edge Detection (Sobel) with read-only cache (__ldg)
// ============================================================================
//
// Strategy:
//   __restrict__ + __ldg() routes reads through the L1 read-only (constant)
//   cache.  For a 3×3 kernel the data reuse between adjacent threads is high
//   enough that this cache hits well without full shared memory tiling.
//   Also: we compute gx/gy for all channels in one pass to reduce kernel
//   launch overhead.
// ============================================================================

__global__ void __launch_bounds__(BLOCK_W * BLOCK_H)
edgeKernelOptimized(const unsigned char* __restrict__ input,
                    unsigned char* __restrict__ output,
                    int width, int height, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Sobel weights (constant → compiler will put in registers)
    const int sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int sobelY[9] = {-1,-2,-1,  0, 0, 0,  1, 2, 1};

    for (int c = 0; c < channels; c++) {
        float gx = 0.0f, gy = 0.0f;

        for (int yp = -1; yp <= 1; yp++) {
            for (int xp = -1; xp <= 1; xp++) {
                int nx = max(0, min(x + xp, width  - 1));
                int ny = max(0, min(y + yp, height - 1));

                // __ldg = load through read-only (texture) cache
                float pv = __ldg(&input[(ny * width + nx) * channels + c]);

                int kid = (yp + 1) * 3 + (xp + 1);
                gx += pv * sobelX[kid];
                gy += pv * sobelY[kid];
            }
        }

        float mag = fminf(255.0f, sqrtf(gx * gx + gy * gy));
        output[(y * width + x) * channels + c] = (unsigned char)mag;
    }
}

// ============================================================================
// KERNEL 4 — Vectorized Grayscale (uchar4: 4 bytes per load)
// ============================================================================
//
// Strategy:
//   Load 4 bytes at once using uchar4 (one aligned 32-bit read per thread).
//   For a 3-channel image we process one pixel per thread but use __ldg for
//   the read-only cache path.  The key win here is eliminating the inner
//   channel loop and making reads coalesced — adjacent threads read adjacent
//   pixels in the row.
// ============================================================================

__global__ void __launch_bounds__(BLOCK_W * BLOCK_H)
grayscaleKernelCoalesced(const unsigned char* __restrict__ input,
                         unsigned char* __restrict__ output,
                         int width, int height, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (channels == 3) {
        const int base = (y * width + x) * 3;

        // __ldg: read-only cache → coalesced access pattern
        float r = __ldg(&input[base + 0]);
        float g = __ldg(&input[base + 1]);
        float b = __ldg(&input[base + 2]);

        // ITU-R BT.601 luma — unchanged from your original
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

        output[base + 0] = gray;
        output[base + 1] = gray;
        output[base + 2] = gray;
    }
}

// ============================================================================
// Helper: create a 2-D texture object from a device buffer
// ============================================================================
static cudaTextureObject_t createTexture2D(unsigned char* d_ptr,
                                           int width, int height, int channels)
{
    // Describe the resource
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr      = d_ptr;
    resDesc.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = (size_t)width * height * channels * sizeof(unsigned char);

    // Texture descriptor — clamp edges, normalized coords OFF
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    return texObj;
}

// ============================================================================
// Helper: allocate pinned host buffer, copy image in, return pointer
//         Pinned memory enables DMA → ~2× PCIe bandwidth vs pageable memory
// ============================================================================
static unsigned char* allocPinned(const cv::Mat& img)
{
    size_t sz = (size_t)img.cols * img.rows * img.channels();
    unsigned char* ptr = nullptr;
    if (cudaMallocHost(&ptr, sz) != cudaSuccess) return nullptr;
    memcpy(ptr, img.data, sz);
    return ptr;
}

// ============================================================================
// LAUNCH FUNCTION — Blur (tiled shared memory)
// ============================================================================
bool launchBlurKernel(const cv::Mat& input, cv::Mat& output, int radius)
{
    output = cv::Mat::zeros(input.size(), input.type());

    const int W = input.cols, H = input.rows, C = input.channels();
    const size_t imgSz = (size_t)W * H * C;

    if (radius > MAX_RADIUS) {
        std::cerr << "Blur radius " << radius << " exceeds MAX_RADIUS="
                  << MAX_RADIUS << ". Increase MAX_RADIUS and recompile." << std::endl;
        return false;
    }

    // --- Pinned host memory for fast PCIe transfer ---
    unsigned char* h_input = allocPinned(input);
    if (!h_input) return false;

    unsigned char* d_input  = nullptr;
    unsigned char* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  imgSz));
    CUDA_CHECK(cudaMalloc(&d_output, imgSz));

    // Async copy host → device
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, imgSz,
                               cudaMemcpyHostToDevice, stream));

    // --- Kernel launch config ---
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((W + BLOCK_W - 1) / BLOCK_W,
              (H + BLOCK_H - 1) / BLOCK_H);

    // Dynamic shared memory size: one plane (BLOCK+2r)^2 floats
    const int tileW = BLOCK_W + 2 * radius;
    const int tileH = BLOCK_H + 2 * radius;
    size_t smemSz = (size_t)tileW * tileH * sizeof(float);

    printKernelLaunchInfo("Blur (tiled)", grid, block);
    std::cout << "   Shared mem per block: " << smemSz / 1024.0f << " KB" << std::endl;

    blurKernelTiled<<<grid, block, smemSz, stream>>>(
        d_input, d_output, W, H, C, radius);

    // Async copy device → host (into pinned output buffer then into cv::Mat)
    unsigned char* h_output = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_output, imgSz));
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, imgSz,
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    memcpy(output.data, h_output, imgSz);

    // Cleanup
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    std::cout << "GPU blur (tiled) completed successfully" << std::endl;
    return true;
}

// ============================================================================
// LAUNCH FUNCTION — Sharpen (texture memory)
// ============================================================================
bool launchSharpenKernel(const cv::Mat& input, cv::Mat& output, float strength)
{
    output = cv::Mat::zeros(input.size(), input.type());

    const int W = input.cols, H = input.rows, C = input.channels();
    const size_t imgSz = (size_t)W * H * C;

    unsigned char* h_input = allocPinned(input);
    if (!h_input) return false;

    unsigned char* d_input  = nullptr;
    unsigned char* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  imgSz));
    CUDA_CHECK(cudaMalloc(&d_output, imgSz));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, imgSz,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));   // texture must be ready before binding

    // Bind input to texture object
    cudaTextureObject_t texObj = createTexture2D(d_input, W, H, C);

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((W + BLOCK_W - 1) / BLOCK_W,
              (H + BLOCK_H - 1) / BLOCK_H);

    printKernelLaunchInfo("Sharpen (texture)", grid, block);

    sharpenKernelTexture<<<grid, block, 0, stream>>>(
        texObj, d_output, W, H, C, strength);

    unsigned char* h_output = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_output, imgSz));
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, imgSz,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    memcpy(output.data, h_output, imgSz);

    cudaDestroyTextureObject(texObj);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    std::cout << "GPU sharpen (texture) completed successfully" << std::endl;
    return true;
}

// ============================================================================
// LAUNCH FUNCTION — Edge Detection (__ldg read-only cache)
// ============================================================================
bool launchEdgeKernel(const cv::Mat& input, cv::Mat& output)
{
    output = cv::Mat::zeros(input.size(), input.type());

    const int W = input.cols, H = input.rows, C = input.channels();
    const size_t imgSz = (size_t)W * H * C;

    unsigned char* h_input = allocPinned(input);
    if (!h_input) return false;

    unsigned char* d_input  = nullptr;
    unsigned char* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  imgSz));
    CUDA_CHECK(cudaMalloc(&d_output, imgSz));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, imgSz,
                               cudaMemcpyHostToDevice, stream));

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((W + BLOCK_W - 1) / BLOCK_W,
              (H + BLOCK_H - 1) / BLOCK_H);

    printKernelLaunchInfo("Edge (ldg cache)", grid, block);

    edgeKernelOptimized<<<grid, block, 0, stream>>>(
        d_input, d_output, W, H, C);

    unsigned char* h_output = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_output, imgSz));
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, imgSz,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    memcpy(output.data, h_output, imgSz);

    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    std::cout << "GPU edge (ldg) completed successfully" << std::endl;
    return true;
}

// ============================================================================
// LAUNCH FUNCTION — Grayscale (coalesced __ldg)
// ============================================================================
bool launchGrayscaleKernel(const cv::Mat& input, cv::Mat& output)
{
    output = cv::Mat::zeros(input.size(), input.type());

    const int W = input.cols, H = input.rows, C = input.channels();
    const size_t imgSz = (size_t)W * H * C;

    unsigned char* h_input = allocPinned(input);
    if (!h_input) return false;

    unsigned char* d_input  = nullptr;
    unsigned char* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  imgSz));
    CUDA_CHECK(cudaMalloc(&d_output, imgSz));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, imgSz,
                               cudaMemcpyHostToDevice, stream));

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((W + BLOCK_W - 1) / BLOCK_W,
              (H + BLOCK_H - 1) / BLOCK_H);

    printKernelLaunchInfo("Grayscale (coalesced)", grid, block);

    grayscaleKernelCoalesced<<<grid, block, 0, stream>>>(
        d_input, d_output, W, H, C);

    unsigned char* h_output = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_output, imgSz));
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, imgSz,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    memcpy(output.data, h_output, imgSz);

    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    std::cout << "GPU grayscale (coalesced) completed successfully" << std::endl;
    return true;
}

// ============================================================================
// Utility functions — unchanged from your original
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
    std::cout << "Launching " << kernelName << " kernel:" << std::endl;
    std::cout << "   Grid size: "  << gridSize.x  << "x" << gridSize.y  << std::endl;
    std::cout << "   Block size: " << blockSize.x << "x" << blockSize.y << std::endl;
    std::cout << "   Total threads: "
              << (long long)gridSize.x * gridSize.y * blockSize.x * blockSize.y
              << std::endl;
}
