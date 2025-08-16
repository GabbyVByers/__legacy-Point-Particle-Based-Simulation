
#include "opengl.h"
#include "state.h"



int main()
{
    InteropOpenGL OpenGL(1200, 800, "Cuda OpenGL Interop", true);
    OpenGL.disableVSYNC();

    SharedArray<Particle> particles;
    for (int i = 0; i < 1000; i++)
    {
        Particle particle;
        particle.position = randVec2f(-1.0f, 1.0f);
        Vec2f vel;
        while (true)
        {
            vel = randVec2f(-1.0f, 1.0f);
            if (length(vel) < 1.0f)
                break;
        }
        normalize(vel);
        vel = vel * 0.001f;
        particle.velocity = vel;
        particles.add(particle);
    }
    particles.updateHostToDevice();

    while (OpenGL.isAlive())
    {
        OpenGL.executePixelKernel(particles.size, particles);
        OpenGL.renderFullScreenQuad();
        OpenGL.renderImGui();
        OpenGL.processUserInput();
        OpenGL.swapBuffers();
    }

    return 0;
}

