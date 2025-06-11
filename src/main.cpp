#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <string>
#include "ImageProcessor.h"

void printUsage(const char* program) {
    std::cout << "ImageFilterGPU-Okab - GPU Image Processing\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program << " <input> <output> <filter> [params] [--cpu]\n";
    std::cout << "  " << program << " test [--cpu]\n\n";
    std::cout << "Filters:\n";
    std::cout << "  blur [radius]      - Gaussian blur (default radius: 5)\n";
    std::cout << "  sharpen [strength] - Image sharpening (default: 1.5)\n";
    std::cout << "  edge               - Edge detection\n";
    std::cout << "  grayscale          - Convert to grayscale\n\n";
    std::cout << "Options:\n";
    std::cout << "  --cpu              - Force CPU processing (for benchmarking)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " input.jpg output.jpg blur 10\n";
    std::cout << "  " << program << " input.jpg output.jpg blur 10 --cpu\n";
    std::cout << "  " << program << " input.jpg output.jpg sharpen 2.0\n";
    std::cout << "  " << program << " input.jpg output.jpg grayscale\n";
    std::cout << "  " << program << " test\n";
    std::cout << "  " << program << " test --cpu\n";
}

int main(int argc, char* argv[]) {
    std::cout << "ImageFilterGPU-Okab v1.0\n";
    std::cout << "OpenCV: " << CV_VERSION << std::endl;
    
    // Check for CPU-only flag FIRST
    bool forceCPU = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--cpu") {
            forceCPU = true;
            std::cout << "CPU-only mode enabled\n";
            break;
        }
    }
    
    // Remove --cpu from argument list for easier parsing
    std::vector<std::string> args;
    for (int i = 0; i < argc; i++) {
        if (std::string(argv[i]) != "--cpu") {
            args.push_back(argv[i]);
        }
    }
    
    // Now use cleaned argument list
    int cleanArgc = args.size();
    
    // Check CUDA availability
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices found: " << deviceCount << std::endl;
    
    if (deviceCount > 0 && !forceCPU) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "GPU Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "Mode: GPU processing\n";
    } else if (forceCPU) {
        std::cout << "Mode: CPU processing (forced)\n";
    } else {
        std::cout << "Mode: CPU processing (no CUDA)\n";
    }
    
    // Check arguments
    if (cleanArgc < 2) {
        printUsage(argv[0]);
        return -1;
    }
    
    std::string command = args[1];
    
    // Test mode
    if (command == "test") {
        std::cout << "\nRunning tests...\n";
        ImageProcessor processor(forceCPU);
        return processor.runTests() ? 0 : -1;
    }
    
    // Normal processing mode
    if (cleanArgc < 4) {
        printUsage(argv[0]);
        return -1;
    }
    
    std::string inputPath = args[1];
    std::string outputPath = args[2];
    std::string filterType = args[3];
    
    // Optional parameters
    float param = 0.0f;
    if (cleanArgc > 4) {
        try {
            param = std::stof(args[4]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid parameter '" << args[4] << "'. Expected a number." << std::endl;
            return -1;
        }
    }
        
    // Load input image
    std::cout << "\nLoading: " << inputPath << std::endl;
    cv::Mat inputImage = cv::imread(inputPath);
    if (inputImage.empty()) {
        std::cerr << "Error: Cannot load image: " << inputPath << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << inputImage.cols << "x" << inputImage.rows 
              << " (" << inputImage.channels() << " channels)" << std::endl;
    
    // Process image
    ImageProcessor processor(forceCPU);  // Pass CPU flag to processor
    cv::Mat outputImage;
    
    std::cout << "Applying " << filterType << " filter..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    bool success = processor.applyFilter(inputImage, outputImage, filterType, param);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    if (!success) {
        std::cerr << "Error: Filter processing failed!" << std::endl;
        return -1;
    }
    
    // Save result
    std::cout << "Saving result: " << outputPath << std::endl;
    if (!cv::imwrite(outputPath, outputImage)) {
        std::cerr << "Error: Cannot save image: " << outputPath << std::endl;
        return -1;
    }
    
    // Success message
    std::cout << "Processing complete!" << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Speed: " << (inputImage.cols * inputImage.rows) / (duration.count() * 1000.0) 
              << " megapixels/second" << std::endl;
    
    return 0;
}