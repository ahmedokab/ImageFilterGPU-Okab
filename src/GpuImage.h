#ifndef GPU_IMAGE_H
#define GPU_IMAGE_H

/*
 * GpuImage.h
 * Ahmed Okab — ImageFilterGPU-Okab
 *
 * Keeps an image resident on the GPU across multiple filter passes.
 * Instead of uploading and downloading for every filter call, you:
 *
 *   1. Upload once   (GpuImage::upload)
 *   2. Apply filters (GpuImage::applyBlur, applySharpen, etc.)
 *      Each filter reads from d_buf[current] and writes to d_buf[1-current],
 *      then flips current. Zero host<->device transfers between filters.
 *   3. Download once (GpuImage::download)
 *
 * Transfer cost for N filters:
 *   Individual mode:  2 * N transfers  (current approach)
 *   Pipeline mode:    2 transfers       (this class)
 */

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

// Forward declarations of kernel launch functions from CudaKernels.cu
// We call the raw kernels directly here so we can pass device pointers
// without going through the cv::Mat path (which would force a transfer).
bool launchBlurKernelRaw     (unsigned char* d_in, unsigned char* d_out,
                               int width, int height, int channels, int radius);
bool launchSharpenKernelRaw  (unsigned char* d_in, unsigned char* d_out,
                               int width, int height, int channels, float strength);
bool launchEdgeKernelRaw     (unsigned char* d_in, unsigned char* d_out,
                               int width, int height, int channels);
bool launchGrayscaleKernelRaw(unsigned char* d_in, unsigned char* d_out,
                               int width, int height);

class GpuImage {
public:
    GpuImage() : d_buf{nullptr, nullptr}, width_(0), height_(0),
                 channels_(0), current_(0), allocated_(false) {}

    ~GpuImage() { free(); }

    // Upload image from host to GPU. Allocates two device buffers (ping-pong).
    bool upload(const cv::Mat& img) {
        if (img.empty() || img.channels() != 3) {
            std::cerr << "[GpuImage] Invalid input image\n";
            return false;
        }

        free();

        width_    = img.cols;
        height_   = img.rows;
        channels_ = img.channels();
        size_     = (size_t)width_ * height_ * channels_;

        if (cudaMalloc(&d_buf[0], size_) != cudaSuccess) return false;
        if (cudaMalloc(&d_buf[1], size_) != cudaSuccess) {
            cudaFree(d_buf[0]); d_buf[0] = nullptr; return false;
        }

        current_   = 0;
        allocated_ = true;

        if (cudaMemcpy(d_buf[0], img.data, size_,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            free(); return false;
        }

        std::cout << "[GpuImage] Uploaded " << width_ << "x" << height_
                  << " (" << size_ / (1024.0 * 1024.0) << " MB) to GPU\n";
        return true;
    }

    // Sync all queued kernels then copy result back to host.
    bool download(cv::Mat& out) const {
        if (!allocated_) return false;
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        out = cv::Mat::zeros(height_, width_, CV_8UC3);
        if (cudaMemcpy(out.data, d_buf[current_], size_,
                       cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        std::cout << "[GpuImage] Downloaded result from GPU\n";
        return true;
    }

    // Apply blur in-place on device (ping-pong between d_buf[0] and d_buf[1])
    bool applyBlur(int radius) {
        if (!allocated_) return false;
        int next = 1 - current_;
        std::cout << "[GpuImage] Pipeline: blur (radius=" << radius << ")\n";
        if (!launchBlurKernelRaw(d_buf[current_], d_buf[next],
                                  width_, height_, channels_, radius))
            return false;
        current_ = next;
        return true;
    }

    bool applySharpen(float strength) {
        if (!allocated_) return false;
        int next = 1 - current_;
        std::cout << "[GpuImage] Pipeline: sharpen (strength=" << strength << ")\n";
        if (!launchSharpenKernelRaw(d_buf[current_], d_buf[next],
                                     width_, height_, channels_, strength))
            return false;
        current_ = next;
        return true;
    }

    bool applyEdge() {
        if (!allocated_) return false;
        int next = 1 - current_;
        std::cout << "[GpuImage] Pipeline: edge detection\n";
        if (!launchEdgeKernelRaw(d_buf[current_], d_buf[next],
                                  width_, height_, channels_))
            return false;
        current_ = next;
        return true;
    }

    bool applyGrayscale() {
        if (!allocated_) return false;
        int next = 1 - current_;
        std::cout << "[GpuImage] Pipeline: grayscale\n";
        if (!launchGrayscaleKernelRaw(d_buf[current_], d_buf[next],
                                       width_, height_))
            return false;
        current_ = next;
        return true;
    }

    // Apply a sequence of filters by name, e.g. {"blur:10", "sharpen:2", "edge"}
    bool applySequence(const std::vector<std::string>& steps) {
        for (const auto& step : steps) {
            std::string name  = step;
            std::string param = "";

            size_t colon = step.find(':');
            if (colon != std::string::npos) {
                name  = step.substr(0, colon);
                param = step.substr(colon + 1);
            }

            if (name == "blur") {
                int r = param.empty() ? 5 : std::stoi(param);
                if (!applyBlur(r)) return false;
            } else if (name == "sharpen") {
                float s = param.empty() ? 1.5f : std::stof(param);
                if (!applySharpen(s)) return false;
            } else if (name == "edge") {
                if (!applyEdge()) return false;
            } else if (name == "grayscale") {
                if (!applyGrayscale()) return false;
            } else {
                std::cerr << "[GpuImage] Unknown filter: " << name << "\n";
                return false;
            }
        }
        return true;
    }

    int width()    const { return width_; }
    int height()   const { return height_; }
    int channels() const { return channels_; }
    bool ready()   const { return allocated_; }

private:
    void free() {
        if (d_buf[0]) { cudaFree(d_buf[0]); d_buf[0] = nullptr; }
        if (d_buf[1]) { cudaFree(d_buf[1]); d_buf[1] = nullptr; }
        allocated_ = false;
    }

    unsigned char* d_buf[2];   // ping-pong buffers on device
    int            width_, height_, channels_;
    size_t         size_;
    int            current_;   // which buffer holds the current result
    bool           allocated_;
};

#endif // GPU_IMAGE_H
