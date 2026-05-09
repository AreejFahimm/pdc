// ============================================================================
// optimized_simd_openmp_tiled_convolution.cpp
// V4 – OpenMP + SIMD + Cache-Tiled Separable Gaussian Convolution
// PDC Spring 2026 Project
//
// Compile:
//    g++ -O3 -mavx2 -mfma -fopenmp \
//    optimized_simd_openmp_tiled_convolution.cpp -o v4
//
// Run:
//    ./v4 [threads]
//
// Example:
//    ./v4 8
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <immintrin.h>
#include <omp.h>

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

    // Normalize
    for (float& v : kernel)
        v /= sum;
}

// ============================================================================
// SIMD Horizontal Pass (AVX2)
// ============================================================================
void horizontalPassSIMD(const float* input,
                        float* temp,
                        int width,
                        int height,
                        const float* kernel,
                        int radius)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < height; r++) {

        const float* src = input + r * width;
              float* dst = temp  + r * width;

        int c = 0;

        // SIMD vectorized region
        for (; c <= width - 8; c += 8) {

            __m256 sum =
                _mm256_setzero_ps();

            for (int k = -radius; k <= radius; k++) {

                float buffer[8];

                for (int lane = 0; lane < 8; lane++) {

                    int cc =
                        std::max(0,
                        std::min(width - 1,
                        c + lane + k));

                    buffer[lane] = src[cc];
                }

                __m256 pixels =
                    _mm256_loadu_ps(buffer);

                __m256 weights =
                    _mm256_set1_ps(kernel[k + radius]);

                // Fused Multiply Add
                sum =
                    _mm256_fmadd_ps(
                        pixels,
                        weights,
                        sum
                    );
            }

            _mm256_storeu_ps(dst + c, sum);
        }

        // Remaining scalar pixels
        for (; c < width; c++) {

            float sum = 0.0f;

            for (int k = -radius; k <= radius; k++) {

                int cc =
                    std::max(0,
                    std::min(width - 1,
                    c + k));

                sum +=
                    src[cc] *
                    kernel[k + radius];
            }

            dst[c] = sum;
        }
    }
}

// ============================================================================
// Cache-Tiled Matrix Transpose
// Improves column locality
// ============================================================================
void transposeTiled(const float* input,
                    float* output,
                    int width,
                    int height)
{
    const int TILE = 32;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < height; i += TILE) {

        for (int j = 0; j < width; j += TILE) {

            int iMax =
                std::min(i + TILE, height);

            int jMax =
                std::min(j + TILE, width);

            for (int ii = i; ii < iMax; ii++) {

                for (int jj = j; jj < jMax; jj++) {

                    output[jj * height + ii] =
                        input[ii * width + jj];
                }
            }
        }
    }
}

// ============================================================================
// Full Optimized Separable Convolution
// ============================================================================
void optimizedConvolution(const float* input,
                          float* temp,
                          float* output,
                          float* transposed1,
                          float* transposed2,
                          int width,
                          int height,
                          const float* kernel,
                          int radius)
{
    // ============================================================
    // Pass 1: Horizontal SIMD Blur
    // ============================================================
    horizontalPassSIMD(
        input,
        temp,
        width,
        height,
        kernel,
        radius
    );

    // ============================================================
    // Transpose for cache-friendly vertical pass
    // ============================================================
    transposeTiled(
        temp,
        transposed1,
        width,
        height
    );

    // ============================================================
    // Pass 2: Horizontal SIMD Blur on transposed image
    // (equivalent to vertical blur)
    // ============================================================
    horizontalPassSIMD(
        transposed1,
        transposed2,
        height,
        width,
        kernel,
        radius
    );

    // ============================================================
    // Final transpose back
    // ============================================================
    transposeTiled(
        transposed2,
        output,
        height,
        width
    );
}

// ============================================================================
// Main Driver
// ============================================================================
int main(int argc, char* argv[])
{
    // =====================================================================
    // User-controlled thread count
    // =====================================================================
    int NUM_THREADS =
        (argc > 1)
        ? std::atoi(argv[1])
        : omp_get_max_threads();

    omp_set_num_threads(NUM_THREADS);

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
    std::cout << " OpenMP + SIMD + Tiling Convolution (V4)\n";
    std::cout << "=====================================================\n";

    std::cout << "Image Size    : "
              << WIDTH << " x " << HEIGHT << "\n";

    std::cout << "Kernel Radius : "
              << RADIUS << "\n";

    std::cout << "OMP Threads   : "
              << NUM_THREADS << "\n";

    std::cout << "SIMD Width    : AVX2 (8 floats)\n";

    std::cout << "Tile Size     : 32 x 32\n\n";

    // =====================================================================
    // Allocate Memory
    // =====================================================================
    std::vector<float> input(N);
    std::vector<float> temp(N);
    std::vector<float> output(N);

    std::vector<float> transposed1(N);
    std::vector<float> transposed2(N);

    std::vector<float> kernel;

    // =====================================================================
    // Generate Random Image
    // =====================================================================
    for (int i = 0; i < N; i++) {

        input[i] =
            static_cast<float>(rand() % 256)
            / 255.0f;
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
    optimizedConvolution(
        input.data(),
        temp.data(),
        output.data(),
        transposed1.data(),
        transposed2.data(),
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

        optimizedConvolution(
            input.data(),
            temp.data(),
            output.data(),
            transposed1.data(),
            transposed2.data(),
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
        (pixelsProcessed / 1e6)
        / (avgMs / 1000.0);

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
    // Hardware Analysis Notes
    // =====================================================================

    std::cout << "\nOptimization Summary:\n";

    std::cout << " - OpenMP Parallelism\n";
    std::cout << " - AVX2 SIMD Vectorization\n";
    std::cout << " - FMA Instructions\n";
    std::cout << " - Cache-Aware Tiling\n";
    std::cout << " - Separable Gaussian Filter\n";

    std::cout << "\nExpected Bottleneck:\n";
    std::cout << "Memory bandwidth / cache locality\n";

    std::cout << "\nBenchmark complete.\n";

    return 0;
}