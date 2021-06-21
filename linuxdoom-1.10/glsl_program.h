/*

#ifndef DOOM_GLSL_PROGRAM_H
#define DOOM_GLSL_PROGRAM_H

#include "shader_tools_common.h"
#include "GLSLShader.h"

class GLSLProgram {
        public:
        GLuint program;
        bool linked;
        private:
        GLSLShader* vertex_shader;
        GLSLShader* fragment_shader;
        public:
        GLSLProgram();
        GLSLProgram(GLSLShader* vertex, GLSLShader* fragment);
        void compile();
        void use();
        private:
        void printLinkError(GLuint program);
};

#endif //DOOM_GLSL_PROGRAM_H
*/
