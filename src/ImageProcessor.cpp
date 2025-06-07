#include "ImageProcessor.h"
#include "CudaKernels.h"
#include <iostream>
#include <cuda_runtime.h>

ImageProcessor::ImageProcessor() : cudaAvailable_(false) {
    cudaAvailable_ = initCuda();
    if (cudaAvailable_) {
        logMessage("CUDA initialized successfully");
        printDeviceInfo();
    } else {
        logMessage("CUDA not available - using CPU fallback");
    }
}

ImageProcessor::~ImageProcessor() {
    // Cleanup resources if needed
}

bool ImageProcessor::initCuda() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

void ImageProcessor::printDeviceInfo() const {
    if (!cudaAvailable_) return;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "🔥 CUDA Device Info:" << std::endl;
    std::cout << "   Name: " << prop.name << std::endl;
    std::cout << "   Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "   Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "   Multiprocessors: " << prop.multiProcessorCount << std::endl;
}

bool ImageProcessor::applyFilter(const cv::Mat& input, cv::Mat& output, 
                                const std::string& filterType, float param) {
    if (!validateInput(input)) {
        return false;
    }
    
    logMessage(std::string("Processing with ") + (cudaAvailable_ ? "GPU" : "CPU"));
    
    if (filterType == "blur") {
        int radius = (param > 0) ? static_cast<int>(param) : 5;
        return cudaAvailable_ ? applyBlurGPU(input, output, radius) 
                              : applyBlurCPU(input, output, radius);
    }
    else if (filterType == "sharpen") {
        float strength = (param > 0) ? param : 1.5f;
        return cudaAvailable_ ? applySharpenGPU(input, output, strength)
                              : applySharpenCPU(input, output, strength);
    }
    else if (filterType == "edge") {
        return cudaAvailable_ ? applyEdgeGPU(input, output)
                              : applyEdgeCPU(input, output);
    }
    else {
        std::cerr << "❌ Unknown filter type: " << filterType << std::endl;
        std::cerr << "Available filters: blur, sharpen, edge" << std::endl;
        return false;
    }
}

bool ImageProcessor::runTests() {
    logMessage("Creating test image...");
    
    // Create test image with various patterns
    cv::Mat testImage = cv::Mat::zeros(512, 512, CV_8UC3);
    
    // Add some geometric shapes
    cv::rectangle(testImage, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 255, 255), -1);
    cv::circle(testImage, cv::Point(400, 400), 80, cv::Scalar(0, 255, 0), -1);
    cv::line(testImage, cv::Point(0, 0), cv::Point(512, 512), cv::Scalar(255, 0, 0), 5);
    
    // Add some noise for better testing
    cv::Mat noise = cv::Mat::zeros(testImage.size(), testImage.type());
    cv::randu(noise, cv::Scalar(0, 0, 0), cv::Scalar(50, 50, 50));
    testImage += noise;
    
    cv::imwrite("data/output/test_original.jpg", testImage);
    logMessage("Test image created and saved");
    
    // Test each filter
    cv::Mat result;
    bool allTestsPassed = true;
    
    // Test blur
    std::cout << "🔍 Testing blur filter..." << std::endl;
    if (applyFilter(testImage, result, "blur", 8)) {
        cv::imwrite("data/output/test_blur.jpg", result);
        std::cout << "✅ Blur test passed" << std::endl;
    } else {
        std::cout << "❌ Blur test failed" << std::endl;
        allTestsPassed = false;
    }
    
    // Test sharpen
    std::cout << "🔍 Testing sharpen filter..." << std::endl;
    if (applyFilter(testImage, result, "sharpen", 2.0f)) {
        cv::imwrite("data/output/test_sharpen.jpg", result);
        std::cout << "✅ Sharpen test passed" << std::endl;
    } else {
        std::cout << "❌ Sharpen test failed" << std::endl;
        allTestsPassed = false;
    }
    
    // Test edge detection
    std::cout << "🔍 Testing edge detection..." << std::endl;
    if (applyFilter(testImage, result, "edge")) {
        cv::imwrite("data/output/test_edge.jpg", result);
        std::cout << "✅ Edge detection test passed" << std::endl;
    } else {
        std::cout << "❌ Edge detection test failed" << std::endl;
        allTestsPassed = false;
    }
    
    if (allTestsPassed) {
        std::cout << "🎉 All tests passed! Check data/output/ for results." << std::endl;
    } else {
        std::cout << "⚠️  Some tests failed. Check implementation." << std::endl;
    }
    
    return allTestsPassed;
}

// GPU implementations (call CUDA kernels)
bool ImageProcessor::applyBlurGPU(const cv::Mat& input, cv::Mat& output, int radius) {
    logMessage("Applying GPU blur (radius: " + std::to_string(radius) + ")");
    
    // Try CUDA kernel first
    if (launchBlurKernel(input, output, radius)) {
        return true;
    }
    
    // If CUDA kernel fails, fallback to CPU
    logMessage("GPU kernel failed, falling back to CPU");
    return applyBlurCPU(input, output, radius);
}

bool ImageProcessor::applySharpenGPU(const cv::Mat& input, cv::Mat& output, float strength) {
    logMessage("Applying GPU sharpen (strength: " + std::to_string(strength) + ")");
    
    if (launchSharpenKernel(input, output, strength)) {
        return true;
    }
    
    logMessage("GPU kernel failed, falling back to CPU");
    return applySharpenCPU(input, output, strength);
}

bool ImageProcessor::applyEdgeGPU(const cv::Mat& input, cv::Mat& output) {
    logMessage("Applying GPU edge detection");
    
    if (launchEdgeKernel(input, output)) {
        return true;
    }
    
    logMessage("GPU kernel failed, falling back to CPU");
    return applyEdgeCPU(input, output);
}
// CPU fallback implementations
bool ImageProcessor::applyBlurCPU(const cv::Mat& input, cv::Mat& output, int radius) {
    try {
        logMessage("Applying CPU blur (radius: " + std::to_string(radius) + ")");
        cv::Size kernelSize(radius * 2 + 1, radius * 2 + 1);
        cv::GaussianBlur(input, output, kernelSize, 0);
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "CPU blur error: " << e.what() << std::endl;
        return false;
    }
}

bool ImageProcessor::applySharpenCPU(const cv::Mat& input, cv::Mat& output, float strength) {
    try {
        logMessage("Applying CPU sharpen (strength: " + std::to_string(strength) + ")");
        cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
                         0, -strength, 0,
                         -strength, 4*strength + 1, -strength,
                         0, -strength, 0);
        cv::filter2D(input, output, input.depth(), kernel);
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "CPU sharpen error: " << e.what() << std::endl;
        return false;
    }
}

bool ImageProcessor::applyEdgeCPU(const cv::Mat& input, cv::Mat& output) {
    try {
        logMessage("Applying CPU edge detection");
        cv::Mat gray, edges;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 100, 200);
        cv::cvtColor(edges, output, cv::COLOR_GRAY2BGR);
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "CPU edge detection error: " << e.what() << std::endl;
        return false;
    }
}

// Helper functions
void ImageProcessor::logMessage(const std::string& message) const {
    std::cout << "[ImageProcessor] " << message << std::endl;
}

bool ImageProcessor::validateInput(const cv::Mat& input) const {
    if (input.empty()) {
        std::cerr << "❌ Error: Input image is empty" << std::endl;
        return false;
    }
    
    if (input.channels() != 3) {
        std::cerr << "❌ Error: Only 3-channel (BGR) images supported" << std::endl;
        return false;
    }
    
    return true;
}