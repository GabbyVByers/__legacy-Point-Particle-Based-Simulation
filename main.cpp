
#include "opengl.h"
#include "state.h"
#include "utilities.h"

int main()
{
    InteropOpenGL OpenGL(1200, 800, "Cuda OpenGL Interop", true);
    OpenGL.enableVSYNC();

    GlobalState globalState;
    initParticles(globalState, 1000);

    

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

