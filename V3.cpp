// ============================================================================
// separable_sequential_convolution.cpp
// V3 – Sequential Separable Gaussian Convolution
// PDC Spring 2026 Project
//
// Compile:
//    g++ -O3 separable_sequential_convolution.cpp -o separable_v3
//
// Run:
//    ./separable_v3
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>

// ============================================================================
// Generate 1D Gaussian Kernel
// ============================================================================
void generateGaussianKernel1D(std::vector<float>& kernel,
                              int radius,
                              float sigma)
{
    int size = 2 * radius + 1;

    kernel.resize(size);

    float sum = 0.0f;

    for (int i = -radius; i <= radius; i++) {

        float value =
            std::exp(-(i * i) /
            (2.0f * sigma * sigma));

        kernel[i + radius] = value;

        sum += value;
    }

    // Normalize kernel
    for (float& v : kernel)
        v /= sum;
}

// ============================================================================
// Horizontal Pass
// ============================================================================
void horizontalPass(const float* input,
                    float* temp,
                    int width,
                    int height,
                    const float* kernel,
                    int radius)
{
    for (int r = 0; r < height; r++) {

        for (int c = 0; c < width; c++) {

            float sum = 0.0f;

            for (int k = -radius; k <= radius; k++) {

                int cc =
                    std::max(0,
                    std::min(width - 1, c + k));

                sum +=
                    input[r * width + cc] *
                    kernel[k + radius];
            }

            temp[r * width + c] = sum;
        }
    }
}

// ============================================================================
// Vertical Pass
// ============================================================================
void verticalPass(const float* temp,
                  float* output,
                  int width,
                  int height,
                  const float* kernel,
                  int radius)
{
    for (int r = 0; r < height; r++) {

        for (int c = 0; c < width; c++) {

            float sum = 0.0f;

            for (int k = -radius; k <= radius; k++) {

                int rr =
                    std::max(0,
                    std::min(height - 1, r + k));

                sum +=
                    temp[rr * width + c] *
                    kernel[k + radius];
            }

            output[r * width + c] = sum;
        }
    }
}

// ============================================================================
// Full Separable Convolution
// ============================================================================
void separableConvolution(const float* input,
                          float* temp,
                          float* output,
                          int width,
                          int height,
                          const float* kernel,
                          int radius)
{
    // Horizontal blur
    horizontalPass(
        input,
        temp,
        width,
        height,
        kernel,
        radius
    );

    // Vertical blur
    verticalPass(
        temp,
        output,
        width,
        height,
        kernel,
        radius
    );
}

// ============================================================================
// Main Driver
// ============================================================================
int main()
{
    // =====================================================================
    // Image Parameters
    // =====================================================================
    const int WIDTH  = 2048;
    const int HEIGHT = 2048;

    const int RADIUS = 8;
    const float SIGMA = 4.0f;

    const int N = WIDTH * HEIGHT;

    // =====================================================================
    // Display Info
    // =====================================================================
    std::cout << "=====================================================\n";
    std::cout << " Sequential Separable Gaussian Convolution (V3)\n";
    std::cout << "=====================================================\n";

    std::cout << "Image Size    : "
              << WIDTH << " x " << HEIGHT << "\n";

    std::cout << "Kernel Radius : "
              << RADIUS << "\n";

    std::cout << "1D Kernel Size: "
              << (2 * RADIUS + 1)
              << "\n\n";

    // =====================================================================
    // Allocate Memory
    // =====================================================================
    std::vector<float> input(N);
    std::vector<float> temp(N);
    std::vector<float> output(N);

    std::vector<float> kernel;

    // =====================================================================
    // Generate Random Image
    // =====================================================================
    for (int i = 0; i < N; i++) {

        input[i] =
            static_cast<float>(rand() % 256) / 255.0f;
    }

    // =====================================================================
    // Generate Gaussian Kernel
    // =====================================================================
    generateGaussianKernel1D(
        kernel,
        RADIUS,
        SIGMA
    );

    // =====================================================================
    // Warm-up Run
    // =====================================================================
    separableConvolution(
        input.data(),
        temp.data(),
        output.data(),
        WIDTH,
        HEIGHT,
        kernel.data(),
        RADIUS
    );

    // =====================================================================
    // Timed Benchmark
    // =====================================================================
    const int REPS = 3;

    double totalMs = 0.0;

    for (int rep = 0; rep < REPS; rep++) {

        auto start =
            std::chrono::high_resolution_clock::now();

        separableConvolution(
            input.data(),
            temp.data(),
            output.data(),
            WIDTH,
            HEIGHT,
            kernel.data(),
            RADIUS
        );

        auto end =
            std::chrono::high_resolution_clock::now();

        double ms =
            std::chrono::duration<double,
            std::milli>(end - start).count();

        totalMs += ms;
    }

    double avgMs = totalMs / REPS;

    // =====================================================================
    // Performance Metrics
    // =====================================================================

    double pixelsProcessed =
        static_cast<double>(WIDTH) * HEIGHT;

    double mpixPerSec =
        (pixelsProcessed / 1e6) / (avgMs / 1000.0);

    // Separable convolution:
    // 2 passes × (2R+1) multiply-adds
    long long operationsPerPixel =
        2 * (2 * RADIUS + 1) * 2;

    double totalOps =
        pixelsProcessed * operationsPerPixel;

    double gflops =
        totalOps / (avgMs / 1000.0) / 1e9;

    // =====================================================================
    // Results
    // =====================================================================

    std::cout << std::fixed
              << std::setprecision(2);

    std::cout << "Average Execution Time : "
              << avgMs << " ms\n";

    std::cout << "Throughput             : "
              << mpixPerSec << " MPix/s\n";

    std::cout << "Estimated GFLOPS       : "
              << gflops << " GFLOPS\n";

    // =====================================================================
    // Complexity Comparison
    // =====================================================================

    int naiveOps =
        (2 * RADIUS + 1) *
        (2 * RADIUS + 1);

    int separableOps =
        2 * (2 * RADIUS + 1);

    std::cout << "\nComplexity Reduction:\n";

    std::cout << "Naive 2D Ops/Pixel     : "
              << naiveOps << "\n";

    std::cout << "Separable Ops/Pixel    : "
              << separableOps << "\n";

    std::cout << "Reduction Factor       : "
              << static_cast<float>(naiveOps)
                 / separableOps
              << "x\n";

    std::cout << "\nBenchmark complete.\n";

    return 0;
}