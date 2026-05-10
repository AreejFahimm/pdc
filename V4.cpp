// ============================================================================
// V4 - SIMD + OpenMP + Tiling Gaussian Convolution
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <immintrin.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

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

int main(int argc, char* argv[])
{
    int NUM_THREADS =
        (argc > 1)
        ? atoi(argv[1])
        : omp_get_max_threads();

    omp_set_num_threads(NUM_THREADS);

    const char* IMAGE_PATH = "1024.png";

    int width, height, channels;

    unsigned char* img =
        stbi_load(
            IMAGE_PATH,
            &width,
            &height,
            &channels,
            1
        );

    if (!img) {
        cout << "Failed to load image.\n";
        return -1;
    }

    int N = width * height;

    vector<float> input(N);
    vector<float> temp(N);
    vector<float> output(N);
    vector<float> transposed1(N);
    vector<float> transposed2(N);

    for (int i = 0; i < N; i++) {
        input[i] = img[i] / 255.0f;
    }

    stbi_image_free(img);


    //     const int RADII[]  = {1, 2, 3, 5};
// const float SIGMAS[] = {0.8f, 1.2f, 1.8f, 2.8f};

    const int RADIUS = 5;
    const float SIGMA = 2.8f;

    vector<float> kernel;

    generateGaussianKernel1D(
        kernel,
        RADIUS,
        SIGMA
    );

    auto start =
        chrono::high_resolution_clock::now();

    horizontalPassSIMD(
        input.data(),
        temp.data(),
        width,
        height,
        kernel.data(),
        RADIUS
    );

    transposeTiled(
        temp.data(),
        transposed1.data(),
        width,
        height
    );

    horizontalPassSIMD(
        transposed1.data(),
        transposed2.data(),
        height,
        width,
        kernel.data(),
        RADIUS
    );

    transposeTiled(
        transposed2.data(),
        output.data(),
        height,
        width
    );

    auto end =
        chrono::high_resolution_clock::now();

    double ms =
        chrono::duration<double, milli>
        (end - start).count();

    cout << fixed << setprecision(2);
    cout << "Execution Time: " << ms << " ms\n";

    vector<unsigned char> outImage(N);

    for (int i = 0; i < N; i++) {

        float v =
            max(0.0f,
            min(1.0f, output[i]));

        outImage[i] =
            static_cast<unsigned char>(v * 255.0f);
    }

    stbi_write_png(
        "output_v4.png",
        width,
        height,
        1,
        outImage.data(),
        width
    );

    cout << "Saved: output_v4.png\n";

    return 0;
}