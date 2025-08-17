
#include "opengl.h"
#include "sharedarray.h"
#include "vec2f.h"

__global__ void particlePhysicsKernel(GlobalState globalState)
{
    int partIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (partIndex >= globalState.particles.size) return;

    Particle& part = globalState.particles.devicePointer[partIndex];
    int numParts = globalState.particles.size;
    
    for (int i = 0; i < numParts; i++)
    {
        if (i == partIndex) continue;
        Particle& other = globalState.particles.devicePointer[i];

        Vec2f acc = other.pos - part.pos;
        float dist = length(acc);
        normalize(acc);
        acc = acc * globalState.gravConstant * (1.0f / (dist * dist));
        part.vel = part.vel + acc;
    }

    if (globalState.isMouseLeft)
    {
        Vec2f mouseAcc = globalState.mousePos - part.pos;
        normalize(mouseAcc);
        mouseAcc = mouseAcc * globalState.mouseAttraction;
        part.vel = part.vel + mouseAcc;
    }
    if (globalState.isMouseRight)
        part.vel = part.vel * globalState.velocityDampening;
    part.pos = part.pos + part.vel;
}

__global__ void renderParticlesKernel(GlobalState globalState)
{
    int partIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (partIndex >= globalState.particles.size) return;

    Particle& part = globalState.particles.devicePointer[partIndex];
    float u = part.pos.x;
    float v = part.pos.y;
    int x = (((u * (globalState.height / (float)globalState.width)) + 1.0f) / 2.0f) * globalState.width;
    int y = ((v + 1.0f) / 2.0f) * globalState.height;
    int pixelIndex = y * globalState.width + x;

    if ((x < 0) || (x >= globalState.width)) return;
    if ((y < 0) || (y >= globalState.height)) return;

    globalState.pixels[pixelIndex] = make_uchar4(255, 255, 255, 255);
}

__global__ void clearScreenKernel(GlobalState globalState)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = -1;
    if ((x < globalState.width) && (y < globalState.height))
        globalState.pixels[y * globalState.width + x] = make_uchar4(0, 0, 0, 255);
}

void InteropOpenGL::executeCudaKernels(GlobalState& globalState)
{
    size_t size = 0;
    globalState.pixels = nullptr;
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&globalState.pixels, &size, cudaPBO);

    globalState.width = screenWidth;
    globalState.height = screenHeight;
    clearScreenKernel <<<WINDOW_blocksPerGrid, WINDOW_threadsPerBlock >>> (globalState);
    cudaDeviceSynchronize();

    int threadsPerBlock = 256;
    int blocksPerGrid = (globalState.particles.size + threadsPerBlock - 1) / threadsPerBlock;
    particlePhysicsKernel <<<blocksPerGrid, threadsPerBlock>>> (globalState);
    cudaDeviceSynchronize();
    renderParticlesKernel <<<blocksPerGrid, threadsPerBlock>>> (globalState);
    cudaDeviceSynchronize();
}

