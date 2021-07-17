#include "r_opengl.h"
#include <GL/glew.h>
#include <stdio.h>

void R_CheckForGlErrors(void) {
    while (1) {
        const GLenum err = glGetError();
        if (err == GL_NO_ERROR) {
            break;
        }

        fprintf(stderr, "GL Error %s", gluErrorString(err));
    }
}

