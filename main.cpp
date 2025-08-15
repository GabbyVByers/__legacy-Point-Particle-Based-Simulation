
#include "sharedarray.h"
#include "particle.h"
#include "opengl.h"

float randf(float min, float max)
{
    return ((rand() / (float)RAND_MAX) * (max - min)) + min;
}

Vec2f randVec2f(float min, float max)
{
    float x = randf(min, max);
    float y = randf(min, max);
    return { x, y };
}

int main()
{
    InteropOpenGL OpenGL(1200, 800, "Cuda OpenGL Interop", false);

    SharedArray<Particle> particles;

    for (int i = 0; i < 1024; i++)
    {
        Particle particle;
        particle.position = randVec2f(-1.0f, 1.0f);
        particle.velocity = randVec2f(-0.01f, 0.01f);
        particles.add(particle);
    }

    particles.updateHostToDevice();

    while (OpenGL.isAlive())
    {

        for (int i = 0; i < particles.size; i++)
        {
            Particle& particle = particles.hostPointer[i];
            particle.position = particle.position + particle.velocity;
        }

        particles.updateHostToDevice();

        OpenGL.executePixelKernel(particles.size, particles);
        OpenGL.renderFullScreenQuad();
        OpenGL.renderImGui();
        OpenGL.swapBuffers();
    }

    return 0;
}

