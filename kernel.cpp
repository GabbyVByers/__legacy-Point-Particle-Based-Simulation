
#include "opengl.h"
#include "sharedarray.h"
#include "vec2f.h"

__global__ void renderParticlesKernel(GlobalState globalState)
{
    int particleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (particleIndex >= globalState.particles.size)
        return;

    Particle& particle = globalState.particles.devicePointer[particleIndex];

    float u = particle.position.x;
    float v = particle.position.y;
    
    int x = (((u * (globalState.height / (float)globalState.width)) + 1.0f) / 2.0f) * globalState.width;
    int y = ((v + 1.0f) / 2.0f) * globalState.height;
    int pixelIndex = y * globalState.width + x;

    if ((x < 0) || (x >= globalState.width))
        return;

    if ((y < 0) || (y >= globalState.height))
        return;

    globalState.pixels[pixelIndex] = make_uchar4(255, 255, 255, 255);
}

__global__ void clearScreenKernel(GlobalState globalState)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = -1;

    if ((x < globalState.width) && (y < globalState.height))
        index = y * globalState.width + x;
    else
        return;

    globalState.pixels[index] = make_uchar4(0, 0, 0, 255);
}

void InteropOpenGL::executeKernels(GlobalState& globalState)
{
    size_t size = 0;
    globalState.pixels = nullptr;
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&globalState.pixels, &size, cudaPBO);

    globalState.width = screenWidth;
    globalState.height = screenHeight;
    clearScreenKernel <<<grid, block>>> (globalState);
    cudaDeviceSynchronize();

    int threadsPerBlock = 256;
    int blocksPerGrid = (globalState.particles.size + threadsPerBlock - 1) / threadsPerBlock;
    renderParticlesKernel <<<blocksPerGrid, threadsPerBlock>>> (globalState);
    cudaDeviceSynchronize();
}

