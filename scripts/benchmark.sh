#!/bin/bash
echo "GPU vs CPU Performance Benchmark"
echo "================================"

IMAGE="data/input/test.png"

echo ""
echo "Testing blur (radius 10):"
echo "  GPU: "
time ./build/ImageFilterGPU-Okab $IMAGE data/output/gpu_test.png blur 10
echo "  CPU: "
time ./build/ImageFilterGPU-Okab $IMAGE data/output/cpu_test.png blur 10 --cpu

echo ""
echo "Testing sharpen (strength 10):"
echo "  GPU: "
time ./build/ImageFilterGPU-Okab $IMAGE data/output/gpu_test.png sharpen 10
echo "  CPU: "
time ./build/ImageFilterGPU-Okab $IMAGE data/output/cpu_test.png sharpen 10 --cpu

echo ""
echo "Testing edge detection:"
echo "  GPU: "
time ./build/ImageFilterGPU-Okab $IMAGE data/output/gpu_test.png edge
echo "  CPU: "
time ./build/ImageFilterGPU-Okab $IMAGE data/output/cpu_test.png edge --cpu

echo ""
echo "Testing grayscale:"
echo "  GPU: "
time ./build/ImageFilterGPU-Okab $IMAGE data/output/gpu_test.png grayscale
echo "  CPU: "
time ./build/ImageFilterGPU-Okab $IMAGE data/output/cpu_test.png grayscale --cpu