// ============================================================================
// combined_gaussian_convolution_benchmark.cpp
//
// V1 -> Sequential Naive 2D
// V2 -> OpenMP Naive 2D
// V3 -> Sequential Separable
// V4 -> OpenMP + SIMD + Tiling Separable
//
// Outputs:
// - Execution Time
// - Throughput
// - GFLOPS
// - Speedup w.r.t Baseline (V1)
//
// Compile:
// g++ -O3 -mavx2 -mfma -fopenmp combined_gaussian_convolution_benchmark.cpp -o benchmark
//
// Run:
// ./benchmark [threads]
//
// Example:
// ./benchmark 8
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include<functional>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


using namespace std;

// ============================================================================
// Kernel Generators
// ============================================================================

void generateGaussianKernel2D(vector<float>& kernel,
                              int radius,
                              float sigma)
{
    int size = 2 * radius + 1;

    kernel.resize(size * size);

    float sum = 0.0f;

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {

            float value =
                exp(-(x * x + y * y) /
                (2.0f * sigma * sigma));

            kernel[(y + radius) * size + (x + radius)] = value;

            sum += value;
        }
    }

    for (float& v : kernel)
        v /= sum;
}

void generateGaussianKernel1D(vector<float>& kernel,
                              int radius,
                              float sigma)
{
    int size = 2 * radius + 1;

    kernel.resize(size);

    float sum = 0.0f;

    for (int i = -radius; i <= radius; i++) {

        float value =
            exp(-(i * i) /
            (2.0f * sigma * sigma));

        kernel[i + radius] = value;

        sum += value;
    }

    for (float& v : kernel)
        v /= sum;
}

// ============================================================================
// V1 - Sequential Naive 2D
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

            for (int ky = -radius; ky <= radius; ky++) {

                for (int kx = -radius; kx <= radius; kx++) {

                    int rr =
                        max(0,
                        min(height - 1, r + ky));

                    int cc =
                        max(0,
                        min(width - 1, c + kx));

                    float pixel =
                        input[rr * width + cc];

                    float weight =
                        kernel[(ky + radius) * kernelSize
                             + (kx + radius)];

                    sum += pixel * weight;
                }
            }

            output[r * width + c] = sum;
        }
    }
}

// ============================================================================
// V2 - OpenMP Naive 2D
// ============================================================================

void convolution2D_OpenMP(const float* input,
                          float* output,
                          int width,
                          int height,
                          const float* kernel,
                          int radius)
{
    int kernelSize = 2 * radius + 1;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < height; r++) {

        for (int c = 0; c < width; c++) {

            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++) {

                for (int kx = -radius; kx <= radius; kx++) {

                    int rr =
                        max(0,
                        min(height - 1, r + ky));

                    int cc =
                        max(0,
                        min(width - 1, c + kx));

                    float pixel =
                        input[rr * width + cc];

                    float weight =
                        kernel[(ky + radius) * kernelSize
                             + (kx + radius)];

                    sum += pixel * weight;
                }
            }

            output[r * width + c] = sum;
        }
    }
}

// ============================================================================
// V3 - Sequential Separable
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
                    max(0,
                    min(width - 1, c + k));

                sum +=
                    input[r * width + cc] *
                    kernel[k + radius];
            }

            temp[r * width + c] = sum;
        }
    }
}

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
                    max(0,
                    min(height - 1, r + k));

                sum +=
                    temp[rr * width + c] *
                    kernel[k + radius];
            }

            output[r * width + c] = sum;
        }
    }
}

void separableConvolution(const float* input,
                          float* temp,
                          float* output,
                          int width,
                          int height,
                          const float* kernel,
                          int radius)
{
    horizontalPass(
        input,
        temp,
        width,
        height,
        kernel,
        radius
    );

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
// V4 - SIMD + OpenMP + Tiling
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

        for (; c <= width - 8; c += 8) {

            __m256 sum =
                _mm256_setzero_ps();

            for (int k = -radius; k <= radius; k++) {

                float buffer[8];

                for (int lane = 0; lane < 8; lane++) {

                    int cc =
                        max(0,
                        min(width - 1,
                        c + lane + k));

                    buffer[lane] = src[cc];
                }

                __m256 pixels =
                    _mm256_loadu_ps(buffer);

                __m256 weights =
                    _mm256_set1_ps(kernel[k + radius]);

                sum =
                    _mm256_fmadd_ps(
                        pixels,
                        weights,
                        sum
                    );
            }

            _mm256_storeu_ps(dst + c, sum);
        }

        for (; c < width; c++) {

            float sum = 0.0f;

            for (int k = -radius; k <= radius; k++) {

                int cc =
                    max(0,
                    min(width - 1,
                    c + k));

                sum +=
                    src[cc] *
                    kernel[k + radius];
            }

            dst[c] = sum;
        }
    }
}

void transposeTiled(const float* input,
                    float* output,
                    int width,
                    int height)
{
    const int TILE = 32;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < height; i += TILE) {

        for (int j = 0; j < width; j += TILE) {

            int iMax = min(i + TILE, height);
            int jMax = min(j + TILE, width);

            for (int ii = i; ii < iMax; ii++) {

                for (int jj = j; jj < jMax; jj++) {

                    output[jj * height + ii] =
                        input[ii * width + jj];
                }
            }
        }
    }
}

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
    horizontalPassSIMD(
        input,
        temp,
        width,
        height,
        kernel,
        radius
    );

    transposeTiled(
        temp,
        transposed1,
        width,
        height
    );

    horizontalPassSIMD(
        transposed1,
        transposed2,
        height,
        width,
        kernel,
        radius
    );

    transposeTiled(
        transposed2,
        output,
        height,
        width
    );
}

// ============================================================================
// Benchmark Helper
// ============================================================================

double benchmark(function<void()> func)
{
    const int REPS = 3;

    double totalMs = 0.0;

    for (int i = 0; i < REPS; i++) {

        auto start =
            chrono::high_resolution_clock::now();

        func();

        auto end =
            chrono::high_resolution_clock::now();

        double ms =
            chrono::duration<double, milli>
            (end - start).count();

        totalMs += ms;
    }

    return totalMs / REPS;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[])
{
    int NUM_THREADS =
        (argc > 1)
        ? atoi(argv[1])
        : omp_get_max_threads();

    omp_set_num_threads(NUM_THREADS);

    int WIDTH;
int HEIGHT;

const int RADIUS = 8;
const float SIGMA = 4.0f;

    cout << fixed << setprecision(2);

    cout << "=====================================================\n";
    cout << " Combined Gaussian Convolution Benchmark\n";
    cout << "=====================================================\n";

    // cout << "Image Size : "
    //      << WIDTH << " x " << HEIGHT << "\n";

    // cout << "Radius     : "
    //      << RADIUS << "\n";

    // cout << "Threads    : "
    //      << NUM_THREADS << "\n\n";e

    // =====================================================================
    // Memory
    // =====================================================================

vector<float> input;

vector<float> output1;
vector<float> output2;
vector<float> output3;
vector<float> output4;

vector<float> temp;

vector<float> transposed1;
vector<float> transposed2;

    vector<float> kernel2D;
    vector<float> kernel1D;

    // =====================================================================
// Load Input Image
// =====================================================================

const char* IMAGE_PATH = "input.jpeg";

int channels;

unsigned char* img =
    stbi_load(
        IMAGE_PATH,
        &WIDTH,
        &HEIGHT,
        &channels,
        1
    );

if (!img) {

    cout << "Failed to load image.\n";
    return -1;
}

   cout << "Image Size : "
         << WIDTH << " x " << HEIGHT << "\n";

    cout << "Radius     : "
         << RADIUS << "\n";

    cout << "Threads    : "
         << NUM_THREADS << "\n\n";

int N = WIDTH * HEIGHT;

// Resize buffers AFTER loading image
input.resize(N);

output1.resize(N);
output2.resize(N);
output3.resize(N);
output4.resize(N);

temp.resize(N);

transposed1.resize(N);
transposed2.resize(N);

// Convert image to float
for (int i = 0; i < N; i++) {

    input[i] =
        img[i] / 255.0f;
}

stbi_image_free(img);

    generateGaussianKernel2D(
        kernel2D,
        RADIUS,
        SIGMA
    );

    generateGaussianKernel1D(
        kernel1D,
        RADIUS,
        SIGMA
    );

    // =====================================================================
    // Warmups
    // =====================================================================

    convolution2D(
        input.data(),
        output1.data(),
        WIDTH,
        HEIGHT,
        kernel2D.data(),
        RADIUS
    );

    convolution2D_OpenMP(
        input.data(),
        output2.data(),
        WIDTH,
        HEIGHT,
        kernel2D.data(),
        RADIUS
    );

    separableConvolution(
        input.data(),
        temp.data(),
        output3.data(),
        WIDTH,
        HEIGHT,
        kernel1D.data(),
        RADIUS
    );

    optimizedConvolution(
        input.data(),
        temp.data(),
        output4.data(),
        transposed1.data(),
        transposed2.data(),
        WIDTH,
        HEIGHT,
        kernel1D.data(),
        RADIUS
    );

    // =====================================================================
    // Benchmarks
    // =====================================================================

    double v1_ms =
        benchmark([&]() {

            convolution2D(
                input.data(),
                output1.data(),
                WIDTH,
                HEIGHT,
                kernel2D.data(),
                RADIUS
            );
        });

    double v2_ms =
        benchmark([&]() {

            convolution2D_OpenMP(
                input.data(),
                output2.data(),
                WIDTH,
                HEIGHT,
                kernel2D.data(),
                RADIUS
            );
        });

    double v3_ms =
        benchmark([&]() {

            separableConvolution(
                input.data(),
                temp.data(),
                output3.data(),
                WIDTH,
                HEIGHT,
                kernel1D.data(),
                RADIUS
            );
        });

    double v4_ms =
        benchmark([&]() {

            optimizedConvolution(
                input.data(),
                temp.data(),
                output4.data(),
                transposed1.data(),
                transposed2.data(),
                WIDTH,
                HEIGHT,
                kernel1D.data(),
                RADIUS
            );
        });

    // =====================================================================
    // Speedups
    // =====================================================================

    double speedup_v2 = v1_ms / v2_ms;
    double speedup_v3 = v1_ms / v3_ms;
    double speedup_v4 = v1_ms / v4_ms;

    // =====================================================================
    // Results Table
    // =====================================================================

    cout << "=====================================================\n";
    cout << " RESULTS\n";
    cout << "=====================================================\n\n";

    cout << left
         << setw(10) << "Version"
         << setw(18) << "Time (ms)"
         << setw(18) << "Speedup"
         << "\n";

    cout << "-----------------------------------------------------\n";

    cout << setw(10) << "V1"
         << setw(18) << v1_ms
         << setw(18) << "1.00x"
         << "\n";

    cout << setw(10) << "V2"
         << setw(18) << v2_ms
         << setw(18) << speedup_v2
         << "\n";

    cout << setw(10) << "V3"
         << setw(18) << v3_ms
         << setw(18) << speedup_v3
         << "\n";

    cout << setw(10) << "V4"
         << setw(18) << v4_ms
         << setw(18) << speedup_v4
         << "\n";

    cout << "\n=====================================================\n";

    // =====================================================================
// Save V4 Output Image
// =====================================================================

vector<unsigned char> outImage(N);

for (int i = 0; i < N; i++) {

    float v =
        max(0.0f,
        min(1.0f, output4[i]));

    outImage[i] =
        static_cast<unsigned char>(v * 255.0f);
}

stbi_write_png(
    "blurred_output.png",
    WIDTH,
    HEIGHT,
    1,
    outImage.data(),
    WIDTH
);

cout << "\nSaved output image: blurred_output.png\n";

    return 0;
}