
#include "opengl.h"

void InteropOpenGL::processUserInput(GlobalState& globalState)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

