// ============================================================================
// V1 - Sequential Naive 2D Gaussian Convolution
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

// ============================================================================
// Generate 2D Gaussian Kernel
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
// Main
// ============================================================================

int main()
{
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
    vector<float> output(N);

    for (int i = 0; i < N; i++) {
        input[i] = img[i] / 255.0f;
    }

    stbi_image_free(img);


//     const int RADII[]  = {1, 2, 3, 5};
// const float SIGMAS[] = {0.8f, 1.2f, 1.8f, 2.8f};

    const int RADIUS = 3;
    const float SIGMA = 1.8f;

    vector<float> kernel;

    generateGaussianKernel2D(
        kernel,
        RADIUS,
        SIGMA
    );

    auto start =
        chrono::high_resolution_clock::now();

    convolution2D(
        input.data(),
        output.data(),
        width,
        height,
        kernel.data(),
        RADIUS
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
        "output_v1.png",
        width,
        height,
        1,
        outImage.data(),
        width
    );

    cout << "Saved: output_v1.png\n";

    return 0;
}