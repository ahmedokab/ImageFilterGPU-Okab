#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include "ImageProcessor.h"
#include "GpuImage.h"

using hrc = std::chrono::high_resolution_clock;
using ms  = std::chrono::milliseconds;

static long long elapsed(hrc::time_point t0)
{
    return std::chrono::duration_cast<ms>(hrc::now() - t0).count();
}

static void printSep() { std::cout << std::string(68, '-') << "\n"; }

// Split "blur:10,sharpen:2,edge" into {"blur:10", "sharpen:2", "edge"}
static std::vector<std::string> splitFilters(const std::string& s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(tok);
    return out;
}

// ============================================================================
// --benchmark  (single-filter, GPU vs CPU comparison table)
// ============================================================================
static int runBenchmark(const std::string& inputPath)
{
    std::cout << "\nLoading: " << inputPath << "\n";
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) { std::cerr << "Error: cannot load " << inputPath << "\n"; return -1; }

    const double MP = img.cols * img.rows / 1e6;
    std::cout << "Image: " << img.cols << "x" << img.rows
              << " (" << std::fixed << std::setprecision(1) << MP << " MP)\n\n";

    struct Test { std::string name, label; float param; };
    std::vector<Test> tests = {
        {"blur",      "Blur (r=10)",    10.f},
        {"sharpen",   "Sharpen (s=10)", 10.f},
        {"edge",      "Edge",            0.f},
        {"grayscale", "Grayscale",       0.f},
    };

    ImageProcessor gpuProc(false);
    ImageProcessor cpuProc(true);

    printSep();
    std::cout << std::left  << std::setw(16) << "Filter"
              << std::right << std::setw(10) << "GPU(ms)"
              << std::setw(12) << "GPU(MP/s)"
              << std::setw(10) << "CPU(ms)"
              << std::setw(12) << "CPU(MP/s)"
              << std::setw(8)  << "Winner"
              << "Speedup\n";
    printSep();

    long long totalGPU = 0, totalCPU = 0;
    for (auto& t : tests) {
        cv::Mat go, co;
        auto t0 = hrc::now(); gpuProc.applyFilter(img, go, t.name, t.param); long long g = elapsed(t0);
        auto t1 = hrc::now(); cpuProc.applyFilter(img, co, t.name, t.param); long long c = elapsed(t1);
        totalGPU += g; totalCPU += c;
        bool gpuWins = g < c;
        std::cout << std::left  << std::setw(16) << t.label
                  << std::right << std::setw(10) << g
                  << std::setw(12) << std::fixed << std::setprecision(1) << MP/(g/1000.0)
                  << std::setw(10) << c
                  << std::setw(12) << MP/(c/1000.0)
                  << std::setw(8)  << (gpuWins ? "GPU" : "CPU")
                  << std::fixed << std::setprecision(2)
                  << (gpuWins ? (double)c/g : (double)g/c) << "x\n";
        cv::imwrite("data/output/bench_gpu_" + t.name + ".png", go);
        cv::imwrite("data/output/bench_cpu_" + t.name + ".png", co);
    }
    printSep();
    bool gpuWins = totalGPU < totalCPU;
    std::cout << std::left  << std::setw(16) << "TOTAL"
              << std::right << std::setw(10) << totalGPU
              << std::setw(12) << " "
              << std::setw(10) << totalCPU
              << std::setw(12) << " "
              << std::setw(8)  << (gpuWins ? "GPU" : "CPU")
              << std::fixed << std::setprecision(2)
              << (gpuWins ? (double)totalCPU/totalGPU : (double)totalGPU/totalCPU) << "x\n";
    printSep();
    return 0;
}

// ============================================================================
// --benchmark-pipeline  (individual vs pipeline mode comparison)
// ============================================================================
static int runBenchmarkPipeline(const std::string& inputPath)
{
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) { std::cerr << "Error: cannot load " << inputPath << "\n"; return -1; }

    const double MP = img.cols * img.rows / 1e6;
    std::vector<std::string> filters = {"blur:10", "sharpen:10", "edge", "grayscale"};

    // silence all logging
    std::cout.setstate(std::ios::failbit);

    struct FR { std::string label; long long gpuMs, cpuMs; };
    std::vector<FR> perFilter;
    ImageProcessor gpuProc(false);
    ImageProcessor cpuProc(true);

    // warmup: first GPU call pays ~300ms CUDA context init, exclude from results
    { cv::Mat tmp; gpuProc.applyFilter(img, tmp, "grayscale", 0); }

    // per-filter: each filter run independently on same input
    for (auto& f : filters) {
        std::string name = f, param = "";
        size_t col = f.find(':');
        if (col != std::string::npos) { name = f.substr(0, col); param = f.substr(col+1); }
        float p = param.empty() ? 0.f : std::stof(param);
        std::string label = name + (param.empty() ? "" : "(" + param + ")");
        cv::Mat gout, cout2;
        auto ga = hrc::now(); gpuProc.applyFilter(img, gout, name, p); long long g = elapsed(ga);
        auto ca = hrc::now(); cpuProc.applyFilter(img, cout2, name, p); long long c = elapsed(ca);
        perFilter.push_back({label, g, c});
    }

    // pipeline GPU: one upload, all filters on device, one download
    GpuImage gpuImg;
    auto t1 = hrc::now();
    gpuImg.upload(img);
    gpuImg.applySequence(filters);
    cv::Mat pipeResult;
    gpuImg.download(pipeResult);
    long long pipeMs = elapsed(t1);
    cv::imwrite("data/output/pipeline_result.png", pipeResult);

    // CPU chained
    cv::Mat cpuResult = img.clone();
    auto t2 = hrc::now();
    for (auto& f : filters) {
        std::string name = f, param = "";
        size_t col = f.find(':');
        if (col != std::string::npos) { name = f.substr(0, col); param = f.substr(col+1); }
        cv::Mat out;
        cpuProc.applyFilter(cpuResult, out, name, param.empty() ? 0.f : std::stof(param));
        cpuResult = out;
    }
    long long cpuMs = elapsed(t2);
    cv::imwrite("data/output/pipeline_cpu.png", cpuResult);

    std::cout.clear();

    auto printLine = [](const std::string& label, long long gMs, long long cMs) {
        bool wins = gMs < cMs;
        double pct = wins ? ((double)cMs / gMs - 1.0) * 100.0
                          : ((double)gMs / cMs - 1.0) * 100.0;
        std::cout << std::left << std::setw(20) << label
                  << std::fixed << std::setprecision(0) << pct << "% "
                  << (wins ? "faster" : "slower")
                  << "  (" << gMs << "ms GPU vs " << cMs << "ms CPU)\n";
    };

    std::cout << "\n";
    for (auto& r : perFilter) printLine(r.label, r.gpuMs, r.cpuMs);
    std::cout << std::string(48, '-') << "\n";
    printLine("pipeline total", pipeMs, cpuMs);

    return 0;
}

// ============================================================================
// --pipeline  (run a filter sequence and save result)
// ============================================================================
static int runPipeline(const std::string& inputPath,
                       const std::string& outputPath,
                       const std::string& filterStr)
{
    std::cout << "\nLoading: " << inputPath << "\n";
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) { std::cerr << "Error: cannot load " << inputPath << "\n"; return -1; }

    std::vector<std::string> filters = splitFilters(filterStr);
    std::cout << "Pipeline: ";
    for (size_t i = 0; i < filters.size(); i++)
        std::cout << filters[i] << (i+1 < filters.size() ? " -> " : "\n");

    GpuImage gpuImg;
    auto t0 = hrc::now();
    if (!gpuImg.upload(img))              { std::cerr << "Upload failed\n";   return -1; }
    if (!gpuImg.applySequence(filters))   { std::cerr << "Pipeline failed\n"; return -1; }
    cv::Mat result;
    if (!gpuImg.download(result))         { std::cerr << "Download failed\n"; return -1; }
    long long ms = elapsed(t0);

    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "Error: cannot save " << outputPath << "\n"; return -1;
    }

    double MP = img.cols * img.rows / 1e6;
    std::cout << "Saved: " << outputPath << "\n"
              << "Time:  " << ms << " ms  ("
              << std::fixed << std::setprecision(1) << MP/(ms/1000.0) << " MP/s)\n"
              << "Transfers: 2 total (1 upload + 1 download)\n";
    return 0;
}

// ============================================================================
// Usage
// ============================================================================
static void printUsage(const char* p)
{
    std::cout
        << "ImageFilterGPU-Okab\n\n"
        << "Single filter:\n"
        << "  " << p << " <in> <out> <filter> [param] [--cpu]\n\n"
        << "Pipeline mode (all filters on GPU, 2 transfers total):\n"
        << "  " << p << " --pipeline <in> <out> blur:10,sharpen:2,edge,grayscale\n\n"
        << "Benchmarks:\n"
        << "  " << p << " --benchmark          <in>   (single filter, GPU vs CPU)\n"
        << "  " << p << " --benchmark-pipeline <in>   (individual vs pipeline vs CPU)\n\n"
        << "Test suite:\n"
        << "  " << p << " test [--cpu]\n\n"
        << "Filters: blur [radius]  sharpen [strength]  edge  grayscale\n\n"
        << "Examples:\n"
        << "  " << p << " data/input/test.png data/output/blur.png blur 10\n"
        << "  " << p << " --pipeline data/input/test.png data/output/result.png blur:10,edge\n"
        << "  " << p << " --benchmark-pipeline data/input/test.png\n";
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
    std::cout << "ImageFilterGPU-Okab v2.0\nOpenCV: " << CV_VERSION << "\n";

    if (argc < 2) { printUsage(argv[0]); return -1; }

    std::string cmd = argv[1];

    // GPU info
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices: " << deviceCount << "\n";
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name
                  << " (" << prop.totalGlobalMem/(1024*1024) << " MB, "
                  << prop.multiProcessorCount << " SMs)\n";
    }

    // Route to mode
    if (cmd == "--benchmark") {
        if (argc < 3) { std::cerr << "Usage: " << argv[0] << " --benchmark <image>\n"; return -1; }
        return runBenchmark(argv[2]);
    }

    if (cmd == "--benchmark-pipeline") {
        if (argc < 3) { std::cerr << "Usage: " << argv[0] << " --benchmark-pipeline <image>\n"; return -1; }
        return runBenchmarkPipeline(argv[2]);
    }

    if (cmd == "--pipeline") {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0] << " --pipeline <in> <out> <filters>\n"
                      << "Example: --pipeline in.png out.png blur:10,sharpen:2,edge\n";
            return -1;
        }
        return runPipeline(argv[2], argv[3], argv[4]);
    }

    // Single filter mode
    bool forceCPU = false;
    std::vector<std::string> args;
    for (int i = 0; i < argc; i++) {
        if (std::string(argv[i]) == "--cpu") forceCPU = true;
        else args.push_back(argv[i]);
    }
    int n = (int)args.size();

    if (args[1] == "test") {
        ImageProcessor proc(forceCPU);
        return proc.runTests() ? 0 : -1;
    }

    if (n < 4) { printUsage(argv[0]); return -1; }

    std::string inputPath  = args[1];
    std::string outputPath = args[2];
    std::string filterType = args[3];
    float param = (n > 4) ? std::stof(args[4]) : 0.f;

    std::cout << "\nLoading: " << inputPath << "\n";
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) { std::cerr << "Error: cannot load " << inputPath << "\n"; return -1; }
    std::cout << "Image: " << img.cols << "x" << img.rows
              << " (" << img.channels() << " ch) -- "
              << (forceCPU ? "CPU mode" : "GPU mode") << "\n";

    ImageProcessor proc(forceCPU);
    cv::Mat out;
    auto t0 = hrc::now();
    if (!proc.applyFilter(img, out, filterType, param)) {
        std::cerr << "Filter failed\n"; return -1;
    }
    long long ms = elapsed(t0);

    if (!cv::imwrite(outputPath, out)) {
        std::cerr << "Error: cannot save " << outputPath << "\n"; return -1;
    }

    double MP = img.cols * img.rows / 1e6;
    std::cout << "Saved: " << outputPath << "\n"
              << "Time:  " << ms << " ms  ("
              << std::fixed << std::setprecision(1) << MP/(ms/1000.0) << " MP/s)\n";
    return 0;
}