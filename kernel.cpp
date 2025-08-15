
#include "opengl.h"
#include "sharedarray.h"
#include "vec2f.h"

__global__ void particleKernel(uchar4* pixels, int width, int height, SharedArray<Particle> particles)
{
    int particleIndex = threadIdx.x;
    Particle particle = particles.devicePointer[particleIndex];
    float u = particle.position.x;
    float v = particle.position.y;

    int x = (((u * (height / (float)width)) + 1.0f) / 2.0f) * width;
    int y = ((v + 1.0f) / 2.0f) * height;

    if ((x < 0) || (x >= width))
        return;

    if ((y < 0) || (y >= height))
        return;

    int pixelIndex = y * width + x;
    pixels[pixelIndex] = make_uchar4(255, 255, 255, 255);
}

__global__ void pixelKernel(uchar4* pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = ((x / (float)width) * 2.0f - 1.0f) * (width / (float)height);
    float v = (y / (float)height) * 2.0f - 1.0f;

    int index = -1;
    if ((x < width) && (y < height))
        index = y * width + x;
    else
        return;

    pixels[index] = make_uchar4(0, 0, 0, 255);
}

void InteropOpenGL::executePixelKernel(int numParticles, SharedArray<Particle> particles)
{
    uchar4* pixels = nullptr;
    size_t size = 0;
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&pixels, &size, cudaPBO);
    pixelKernel <<<grid, block>>> (pixels, screenWidth, screenHeight);
    particleKernel <<<1, numParticles>>> (pixels, screenWidth, screenHeight, particles);
    cudaDeviceSynchronize();
}

