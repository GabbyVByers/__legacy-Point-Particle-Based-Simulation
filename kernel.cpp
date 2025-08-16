
#include "opengl.h"
#include "sharedarray.h"
#include "vec2f.h"

__global__ void particlePhysicsKernel(GlobalState globalState)
{
    int particleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (particleIndex >= globalState.particles.size)
        return;

    Particle& particle = globalState.particles.devicePointer[particleIndex];
    int numParticles = globalState.particles.size;
    Vec2f& pos = particle.position;
    Vec2f& vel = particle.velocity;

    pos = pos + vel;

    for (int i = 0; i < numParticles; i++)
    {
        if (i == particleIndex)
            continue;

        Particle& other = globalState.particles.devicePointer[i];
        float invDist = 1.0f / distance(pos, other.position);

        Vec2f acceleration = other.position - pos;
        normalize(acceleration);
        acceleration = acceleration * 0.0000001f;

        vel = vel + acceleration;
    }

}

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
        globalState.pixels[y * globalState.width + x] = make_uchar4(0, 0, 0, 255);
}

void InteropOpenGL::executeKernels(GlobalState& globalState)
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

