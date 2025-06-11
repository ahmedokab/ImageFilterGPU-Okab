# ImageFilterGPU-Okab

High-Performance CUDA Image Processing Pipeline (Blur, Sharpen, Grayscale, Edge Detection)

---

## 📊 Project Summary

In this project, I implemented a full image processing pipeline using CUDA and OpenCV for comparison:

- **Filters Implemented:**
  - Gaussian Blur (`blur`)
  - Image Sharpening (`sharpen`)
  - Grayscale Conversion (`grayscale`)
  - Sobel Edge Detection (`edge`)
  
- **Both GPU and CPU modes are supported:**
  - CPU filters use OpenCV's highly optimized internal implementations.
  - GPU filters use custom CUDA kernels that parallelize image processing.

---

## ⚙️ CUDA GPU Info

- NVIDIA GeForce RTX 3060 Laptop GPU
- Compute Capability: 8.6
- 30 Multiprocessors
- 6 GB VRAM

---

##  Results Summary

| Filter Type        | CPU Execution   | GPU Execution   |
| ------------------- | --------------- | ---------------- |
| **Blur (radius 10)** | 43 ms (603.68 MP/s) | 302 ms (85.95 MP/s) |
| **Sharpen (strength 10)** | 108 ms (240.35 MP/s) | 286 ms (90.76 MP/s) |
| **Edge Detection**  | 60 ms (432.64 MP/s) | 257 ms (101.00 MP/s) |
| **Grayscale**       | 17 ms (1526.96 MP/s)  | 231 ms (112.37 MP/s)  |

✅ **CPU outperfoprmed GPU for all filters, due to naivety of GPU implementation I used. This was interesting to me, and I'll make sure to look at ways to optimize GPU usage in the future.**

---

## Filter Comparison Summary

### 1️⃣ Blur (radius 10)

- Smooths image by averaging neighboring pixels.
- Currently uses naive global memory kernel (bottleneck for GPU).

### 2️⃣ Sharpen (strength 10)

- Strength of `10.0` gives clear sharpening effect.
- Kernel applies weighted convolution formula:
- center = 1 + 4 * strength
- neighbors = -strength



### 3️⃣ Edge Detection (Sobel)

- Sobel operator applied in both X and Y directions.
- Outputs edge maps from grayscale input.

### 4️⃣ Grayscale

- Luma-weighted grayscale formula was used:

- Gray = 0.299 * R + 0.587 * G + 0.114 * B applied to the pixels


---

##  Command Line Usage

```bash
# Format:
./ImageFilterGPU-Okab <input_image> <output_image> <filter> [parameter]

# Examples:

# Gaussian Blur (radius 10)
./ImageFilterGPU-Okab data/input/test.png data/output/blur.png blur 10

# Sharpening (strength 10)
./ImageFilterGPU-Okab data/input/test.png data/output/sharpen.png sharpen 10

# Edge Detection
./ImageFilterGPU-Okab data/input/test.png data/output/edge.png edge

# Grayscale Conversion
./ImageFilterGPU-Okab data/input/test.png data/output/grayscale.png grayscale

-Default radiuses used when not provided

## 📚 Key Takeaways & Learnings

- CUDA correctly parallelized all filters across millions of threads.
- Sobel, blur, grayscale, and sharpening filters fully functional.
- Learned CUDA memory model: global memory, shared memory, memory coalescing.
- Implemented complete host-device synchronization to ensure all threads are handled and error handling.
- Successfully handled large high-resolution images (24MP).
- Handled CUDA errors and properly fell back to CPU when exceptions occur. This is done through returning a different value whenever the GPU implementation failed, then falling back to CPU function.

---

## ⚠ Why is CPU still faster than GPU?

- OpenCV uses highly optimized multi-threaded C++ code with SIMD (single intstruction, multiple data) vectorization.
- OpenCV kernels fuse multiple operations (blurring, padding, color conversion) with no extra data transfers.
- My current CUDA kernels:
  - Use naive global memory access (non-coalesced, slow).
  - Perform redundant global reads across threads.
  - Require full host↔device transfers for every filter pass.
  - Launch independent kernels instead of pipelined fused kernels.

---

## 🛠 Future Optimizations that I have researched could be possible

- Shared memory (tiling) for blur, sharpen, edge detection. Through some research, I have foudn out that tiling allows us to minimize the external memory accesses to the GPU during fragment shading. This improves performance drastically.
- Global memory coalescing.
- Kernel fusion (in order to reduce redundant global reads).

---

## 🏁 Conclusion

While my CUDA implementation parallelizes filters properly, the full GPU implementation was not completely harnessed in my opinion and as shown by the data. OpenCV remains faster due to highly optimized CPU-level multi-threading and vectorization. The next step is shared memory optimization, and this will allow my kernels to finally outperform CPU by fully utilizing the CUDA memory hierarchy and reduce global memory overhead. 

This project has developed my C++, CUDA and low-level software development skills, allowing me to understand the memory behind storing and processing images and to manipulate these images. Additionally, this project introduced me to powerful frameworks like OpenCV and CUDA, both of which are highly valuable in modern computer vision, GPU computing, and real-world industry applications.

Credits: Ahmed Okab
