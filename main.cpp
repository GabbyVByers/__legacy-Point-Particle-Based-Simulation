
#include "opengl.h"
#include "state.h"
#include "utilities.h"

int main()
{
    InteropOpenGL OpenGL(1200, 800, "Cuda OpenGL Interop", true);
    OpenGL.enableVSYNC();

    GlobalState globalState;
    initParticles(globalState, 10);

    while (OpenGL.isAlive())
    {
        OpenGL.getMouseProperties(globalState);
        OpenGL.processUserInput(globalState);
        OpenGL.executeCudaKernels(globalState);
        OpenGL.renderFullScreenQuad();
        OpenGL.renderImGui(globalState);
        OpenGL.swapBuffers();
    }

    globalState.particles.free();
    OpenGL.free();

    return 0;
}

