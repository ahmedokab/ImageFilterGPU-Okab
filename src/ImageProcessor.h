#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>

class ImageProcessor {
public:
    ImageProcessor(bool forceCPU = false);
    ~ImageProcessor();
    
    // Main processing function
    bool applyFilter(const cv::Mat& input, cv::Mat& output, 
                    const std::string& filterType, float param = 0.0f);
    
    // Test suite
    bool runTests();
    
    // Utility functions
    bool isCudaAvailable() const { return cudaAvailable_; }
    void printDeviceInfo() const;
    
private:
    bool cudaAvailable_;
    bool forceCPU_;
    
    // Initialization
    bool initCuda();
    
    // GPU filter implementations
    bool applyBlurGPU(const cv::Mat& input, cv::Mat& output, int radius);
    bool applySharpenGPU(const cv::Mat& input, cv::Mat& output, float strength);
    bool applyEdgeGPU(const cv::Mat& input, cv::Mat& output);
    
    //cpu version
    bool applyBlurCPU(const cv::Mat& input, cv::Mat& output, int radius);
    bool applySharpenCPU(const cv::Mat& input, cv::Mat& output, float strength);
    bool applyEdgeCPU(const cv::Mat& input, cv::Mat& output);


    bool applyGrayscaleGPU(const cv::Mat& input, cv::Mat& output);
    bool applyGrayscaleCPU(const cv::Mat& input, cv::Mat& output);
    
    // Helper functions
    void logMessage(const std::string& message) const;
    bool validateInput(const cv::Mat& input) const;
};

#endif // IMAGE_PROCESSOR_H