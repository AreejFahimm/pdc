// ============================================================================
// baseline_sequential_naive_2d_convolution.cpp
// V1 – Sequential Naive 2D Gaussian Convolution
// PDC Spring 2026 Project Baseline Version
//
// Compile:
//    g++ -O2 baseline_sequential_naive_2d_convolution.cpp -o baseline
//
// Run:
//    ./baseline
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>

// ============================================================================
// Generate 2D Gaussian Kernel
// ============================================================================
void generateGaussianKernel2D(std::vector<float>& kernel,
                              int radius,
                              float sigma)
{
    int size = 2 * radius + 1;
    kernel.resize(size * size);

    float sum = 0.0f;

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {

            float value = std::exp(-(x * x + y * y) /
                          (2.0f * sigma * sigma));

            kernel[(y + radius) * size + (x + radius)] = value;
            sum += value;
        }
    }

    // Normalize kernel
    for (float& v : kernel)
        v /= sum;
}

// ============================================================================
// Sequential Naive 2D Convolution
// ============================================================================
void convolution2D(const float* input,
                   float* output,
                   int width,
                   int height,
                   const float* kernel,
                   int radius)
{
    int kernelSize = 2 * radius + 1;

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {

            float sum = 0.0f;

            // Full 2D convolution window
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {

                    int rr = std::max(0, std::min(height - 1, r + ky));
                    int cc = std::max(0, std::min(width - 1,  c + kx));

                    float pixel  = input[rr * width + cc];
                    float weight = kernel[(ky + radius) * kernelSize
                                           + (kx + radius)];

                    sum += pixel * weight;
                }
            }

            output[r * width + c] = sum;
        }
    }
}

// ============================================================================
// Main Driver
// ============================================================================
int main()
{
    // Image dimensions
    const int WIDTH  = 2048;
    const int HEIGHT = 2048;

    // Gaussian radius
    const int RADIUS = 8;
    const float SIGMA = 4.0f;

    const int N = WIDTH * HEIGHT;

    std::cout << "======================================================\n";
    std::cout << " Sequential Naive 2D Gaussian Convolution Baseline\n";
    std::cout << "======================================================\n";
    std::cout << "Image Size : " << WIDTH << " x " << HEIGHT << "\n";
    std::cout << "Kernel Radius : " << RADIUS << "\n";
    std::cout << "Kernel Size : " << (2 * RADIUS + 1)
              << " x " << (2 * RADIUS + 1) << "\n\n";

    // Allocate memory
    std::vector<float> input(N);
    std::vector<float> output(N);
    std::vector<float> kernel;

    // Generate random image
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(rand() % 256) / 255.0f;
    }

    // Generate Gaussian kernel
    generateGaussianKernel2D(kernel, RADIUS, SIGMA);

    // =====================================================================
    // Warm-up run
    // =====================================================================
    convolution2D(input.data(),
                  output.data(),
                  WIDTH,
                  HEIGHT,
                  kernel.data(),
                  RADIUS);

    // =====================================================================
    // Timed execution
    // =====================================================================
    const int REPS = 3;
    double totalMs = 0.0;

    for (int rep = 0; rep < REPS; rep++) {

        auto start = std::chrono::high_resolution_clock::now();

        convolution2D(input.data(),
                      output.data(),
                      WIDTH,
                      HEIGHT,
                      kernel.data(),
                      RADIUS);

        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        totalMs += ms;
    }

    double avgMs = totalMs / REPS;

    // =====================================================================
    // Performance Metrics
    // =====================================================================

    double pixelsProcessed = static_cast<double>(WIDTH) * HEIGHT;

    double mpixPerSec = (pixelsProcessed / 1e6) / (avgMs / 1000.0);

    long long operationsPerPixel =
        (2 * RADIUS + 1) * (2 * RADIUS + 1) * 2;

    double totalOps = pixelsProcessed * operationsPerPixel;

    double gflops = totalOps / (avgMs / 1000.0) / 1e9;

    // =====================================================================
    // Output Results
    // =====================================================================

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Average Execution Time : "
              << avgMs << " ms\n";

    std::cout << "Throughput             : "
              << mpixPerSec << " MPix/s\n";

    std::cout << "Estimated GFLOPS       : "
              << gflops << " GFLOPS\n";

    std::cout << "\nBaseline benchmark complete.\n";

    return 0;
}
