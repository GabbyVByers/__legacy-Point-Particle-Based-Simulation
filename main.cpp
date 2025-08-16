
#include "opengl.h"
#include "state.h"
#include "utilities.h"

int main()
{
    InteropOpenGL OpenGL(1200, 800, "Cuda OpenGL Interop", true);
    OpenGL.enableVSYNC();

    GlobalState globalState;
    initParticles(globalState, 4);

    while (OpenGL.isAlive())
    {
        OpenGL.executeCudaKernels(globalState);
        OpenGL.renderFullScreenQuad();
        OpenGL.renderImGui(globalState);
        OpenGL.processUserInput();
        OpenGL.swapBuffers();
    }

    globalState.particles.free();
    OpenGL.free();

    return 0;
}

