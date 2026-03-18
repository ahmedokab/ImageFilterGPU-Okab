# ImageFilterGPU-Okab

A CUDA image processing pipeline I built to understand how GPUs actually work at the memory level. Implements blur, sharpen, Sobel edge detection, and grayscale on custom CUDA kernels, with a full CPU comparison using OpenCV.

---

## My GPU

- NVIDIA GeForce RTX 3060 Laptop GPU
- Compute Capability 8.6
- 30 Multiprocessors, 6 GB VRAM

---

## The Story

### Phase 1: Naive implementation

I started by writing the simplest possible kernels. Each thread handles one pixel, reads all its neighbors from global memory, and writes the result. Straightforward to write, and I assumed parallelizing across 26 million threads would be fast.

It wasn't.

| Filter | CPU | GPU | Ratio |
|---|---|---|---|
| Blur (radius 10) | 43ms | 748ms | GPU 17x slower |
| Sharpen (strength 10) | 108ms | 342ms | GPU 3x slower |
| Edge Detection | 60ms | 341ms | GPU 6x slower |
| Grayscale | 17ms | 329ms | GPU 19x slower |

GPU lost every single filter by a wide margin. I spent time figuring out exactly why before touching the code.

The problem with blur is the clearest example. At radius=10, each thread reads a 21x21 neighborhood from global memory: 441 reads per pixel. Across a 26MP image that's roughly 11.5 billion global memory reads. Global memory on a GPU has around 200 cycle latency. The threads were spending almost all their time waiting on memory, not computing.

### Phase 2: Kernel optimizations

I rewrote all four kernels with three techniques:

**Shared memory tiling (blur)**

Instead of every thread independently reading from global memory, the whole block cooperates to load a tile into `__shared__` memory first. Shared memory is on-chip and takes about 1 cycle. Then every thread reads its neighborhood from there instead of DRAM.

```
Global reads per block:
  Naive   ->  256 threads x 441 reads  =  112,896 reads
  Tiled   ->  36 x 36 tile load        =    1,296 reads  ->  87x fewer
```

**`__ldg` read-only cache (sharpen, edge, grayscale)**

`__ldg()` routes reads through the L1 read-only cache. For 3x3 stencil kernels where neighboring threads read overlapping neighborhoods, most of those reads hit cache instead of DRAM.

**Loop unrolling (sharpen, edge)**

`#pragma unroll` on the 3x3 loops tells the compiler to emit 9 independent instructions instead of a counted loop. Eliminates branch overhead and lets the compiler schedule instructions better.

Results after optimization:

| Filter | CPU | GPU | Result |
|---|---|---|---|
| Blur (radius 10) | 40ms | 313ms | CPU still faster |
| Sharpen (strength 10) | 111ms | 90ms | GPU wins |
| Edge Detection | 67ms | 120ms | CPU still faster |
| Grayscale | 13ms | 155ms | CPU still faster |

Sharpen beat CPU for the first time. Blur dropped from 748ms to 313ms. But blur, edge, and grayscale were still losing, and I had a clear idea why.

Every filter call copies 78MB to the GPU and 78MB back over PCIe. On WSL2 that costs about 75ms each direction. So before the kernel even starts, you've already paid 150ms in transfer overhead. For grayscale, the kernel itself runs in about 2ms. The math just doesn't work.

### Phase 3: Pipeline mode

The fix was to stop treating each filter as an independent round trip. Instead of uploading and downloading for every filter, I wrote a `GpuImage` class that keeps the image resident on the GPU across all filter passes.

```
Old approach (8 transfers for 4 filters):
  upload -> blur -> download -> upload -> sharpen -> download -> ...

Pipeline approach (2 transfers total):
  upload -> blur -> sharpen -> edge -> grayscale -> download
```

The implementation uses a ping-pong buffer: two device buffers allocated once, and after each kernel the active buffer flips. Blur reads from `d_buf[0]` and writes to `d_buf[1]`, sharpen reads from `d_buf[1]` and writes to `d_buf[0]`, and so on. Data never leaves the GPU between filters.

### Phase 4: Removing per-kernel synchronization

After adding pipeline mode I was still seeing higher latency than expected. The issue was `cudaDeviceSynchronize()` sitting at the end of every raw kernel call. That function makes the CPU wait for the GPU to fully finish before doing anything else, which was effectively serializing every kernel launch with a CPU stall.

In pipeline mode, kernels on the default CUDA stream already execute in order automatically. Sharpen can't start before blur finishes because they're on the same stream and CUDA guarantees that. So the per-kernel syncs were just adding overhead without doing anything useful.

I removed `cudaDeviceSynchronize()` from all four raw kernel functions and moved a single sync to the `download()` call at the end, right before copying the result back to the CPU.

Final results:

| Filter | CPU | GPU | Result |
|---|---|---|---|
| Blur (radius 10) | 39ms | 115ms | CPU faster |
| Sharpen (strength 10) | 101ms | 74ms | GPU 36% faster |
| Edge Detection | 96ms | 78ms | GPU 23% faster |
| Grayscale | 11ms | 85ms | CPU faster |
| Pipeline total | 271ms | 157ms | GPU 73% faster |

Pipeline GPU beat CPU by 73%. Sharpen and edge both win individually now too.

---

## 4 Challenges I Overcame

**1. GPU being slower than CPU**

My first assumption was that parallelism automatically meant speed. The naive results proved that wrong immediately. Figuring out why required understanding the GPU memory hierarchy, calculating actual global memory read counts, and realizing the problem wasn't the parallelism but the memory access pattern. That led directly to the tiling solution.

**2. Transfer overhead killing single-filter performance**

Even after optimizing the kernels, most filters still lost because of PCIe transfer cost. I kept trying to make kernels faster when the bottleneck wasn't the kernel at all. It took actually timing the individual steps (cudaMalloc, cudaMemcpy, kernel, sync, cudaMemcpy back) to see that compute was only 15% of the total time. The fix was architectural, not algorithmic.

**3. Incorrect benchmark results from CUDA context initialization**

Early benchmark runs showed blur taking 580ms when it should have been around 280ms. The first CUDA kernel call in any process pays a one-time ~300ms context initialization cost that has nothing to do with the kernel itself. Without a warmup call before timing, blur was eating that penalty and making the results misleading. Added a throwaway filter call before the timed loop.

**4. Silent serialization from cudaDeviceSynchronize**

Pipeline mode was faster than individual mode but not by as much as expected. I had `cudaDeviceSynchronize()` after every raw kernel call, which was stalling the CPU between each stage of the pipeline even though CUDA's default stream already guarantees sequential execution. Removing those syncs and moving one sync to the download step cut pipeline latency significantly and helped GPU beat CPU overall.

---

## What I Learned

The GPU memory hierarchy is what actually determines performance, not thread count. I launched 26 million threads from day one and it didn't matter because those threads were all waiting on slow global memory. The entire performance gap between the naive and optimized blur comes down to one fact: shared memory has 200x lower latency than global memory.

Measuring matters more than intuition. I assumed GPU would be fast, it wasn't. I assumed kernel optimization was the bottleneck, it was the transfers. I assumed pipeline mode would be dramatically faster, it was okay until I found the synchronization issue. Every assumption was wrong and every fix came from actually measuring where time was going.

The transfer bottleneck is a fundamental constraint of the CPU/GPU architecture, not something you can optimize away at the kernel level. The right solution is to minimize round trips, which is what pipeline mode does. In production ML systems this is why data stays on the GPU across an entire training loop rather than being transferred per operation.

---

## Usage

```bash
# single filter
./ImageFilterGPU-Okab data/input/test.png data/output/blur.png blur 10
./ImageFilterGPU-Okab data/input/test.png data/output/sharpen.png sharpen 10
./ImageFilterGPU-Okab data/input/test.png data/output/edge.png edge
./ImageFilterGPU-Okab data/input/test.png data/output/grayscale.png grayscale

# force CPU
./ImageFilterGPU-Okab data/input/test.png data/output/blur.png blur 10 --cpu

# pipeline: all filters in one GPU session, 2 transfers total
./ImageFilterGPU-Okab --pipeline data/input/test.png data/output/result.png blur:10,sharpen:10,edge,grayscale

# benchmark: GPU vs CPU per filter + pipeline comparison
./ImageFilterGPU-Okab --benchmark-pipeline data/input/test.png
```

---

## Build

```bash
mkdir build && cd build
cmake ..
make
```

Requires CUDA Toolkit, OpenCV 4.x, CMake 3.18+.

---

Ahmed Okab