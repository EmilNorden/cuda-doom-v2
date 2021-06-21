
#ifndef DOOM_GL_WINDOW_H
#define DOOM_GL_WINDOW_H

#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

struct GLFWwindow;

class GlWindow {
        public:
        GlWindow(const std::string& title, int width, int height, GLFWkeyfun key_callback); // TODO: Fix the key_callback. This is a quick and dirty solution that exposes glfw to the rest of the code.

        void draw();
        void swap();

        [[nodiscard]] bool should_close();

        [[nodiscard]] GLFWwindow *handle() { return m_window; } // TODO: Same as above, leaking glfw too much
        private:
        GLuint VBO, VAO, EBO;
        GLFWwindow *m_window;
        int m_width;
        int m_height;
};

#endif //DOOM_GL_WINDOW_H
