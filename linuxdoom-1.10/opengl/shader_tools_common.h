//
// Created by emil on 2021-05-08.
//

#ifndef RENDERER_SHADER_TOOLS_COMMON_H
#define RENDERER_SHADER_TOOLS_COMMON_H

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

#include <GL/glew.h>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>

// Simple helper to switch between character arrays and C++ strings
struct ShaderStringHelper{
    const char *p;
    ShaderStringHelper(const std::string& s) : p(s.c_str()) {}
    operator const char**() { return &p; }
};

// Function to load text from file
// static, we only want this function to be available in this file's scope
inline static std::string loadFileToString(const char *filename){
    std::ifstream file(filename, std::ios::in);
    std::string text;
    if (file){
        file.seekg(0, std::ios::end); // go to end
        text.resize(file.tellg()); // resize text buffer to file size
        file.seekg(0, std::ios::beg); // back to begin
        file.read(&text[0], text.size()); // read into buffer
        file.close();
    }
    else {
        std::string error_message = std::string("File not found: ") + filename;
        std::cerr << error_message.c_str();
        throw std::runtime_error(error_message);
    }
    return text;
}

#endif //RENDERER_SHADER_TOOLS_COMMON_H
