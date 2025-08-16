
#include "opengl.h"
#include "state.h"
#include "utilities.h"

int main()
{
    InteropOpenGL OpenGL(1200, 800, "Cuda OpenGL Interop", true);
    OpenGL.disableVSYNC();

    GlobalState globalState;

    for (int i = 0; i < 1000; i++)
    {
        Particle particle;
        particle.position = randVec2f(-1.0f, 1.0f);
        globalState.particles.add(particle);
    }
    globalState.particles.updateHostToDevice();

    while (OpenGL.isAlive())
    {
        OpenGL.executeKernels(globalState);
        OpenGL.renderFullScreenQuad();
        OpenGL.renderImGui();
        OpenGL.processUserInput();
        OpenGL.swapBuffers();
    }

    globalState.particles.free();
    OpenGL.free();

    return 0;
}

