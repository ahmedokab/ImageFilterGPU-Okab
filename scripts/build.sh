

# Build script

#!/bin/bash
echo "🔨 Building ImageFilterGPU-Okab..."
mkdir -p build
cd build
cmake .. && make -j$(nproc)
if [ -f "ImageFilterGPU-Okab" ]; then
    echo "✅ Build successful!"
    echo "Run: ./build/ImageFilterGPU-Okab"
else
    echo "❌ Build failed!"
fi


