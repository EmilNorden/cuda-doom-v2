//
// Created by emil on 2021-05-08.
//

#ifndef RENDERER_GLSLPROGRAM_H
#define RENDERER_GLSLPROGRAM_H

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
    void use() const;
private:
    void printLinkError(GLuint program);
};

#endif //RENDERER_GLSLPROGRAM_H
