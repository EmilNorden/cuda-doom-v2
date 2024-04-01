// Emacs style mode select   -*- C++ -*- 
//-----------------------------------------------------------------------------
//
// $Id:$
//
// Copyright (C) 1993-1996 by id Software, Inc.
//
// This source is available for distribution and/or modification
// only under the terms of the DOOM Source Code License as
// published by id Software. All rights reserved.
//
// The source is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// FITNESS FOR A PARTICULAR PURPOSE. See the DOOM Source Code License
// for more details.
//
// $Log:$
//
// DESCRIPTION:
//	Gamma correction LUT stuff.
//	Functions to draw patches (by post) directly to screen.
//	Functions to blit a block to the screen.
//
//-----------------------------------------------------------------------------

#include <sys/time.h>
#include <time.h>
#include <SDL2/SDL.h>

static const char
        rcsid[] = "$Id: v_video.c,v 1.5 1997/02/03 22:45:13 b1 Exp $";


static const char *glsl_drawtex_vertshader_src =
        "#version 330 core\n"
        "layout (location = 0) in vec3 position;\n"
        "layout (location = 1) in vec3 color;\n"
        "layout (location = 2) in vec2 texCoord;\n"
        "\n"
        "out vec3 ourColor;\n"
        "out vec2 ourTexCoord;\n"
        "\n"
        "void main()\n"
        "{\n"
        "	gl_Position = vec4(position, 1.0f);\n"
        "	ourColor = color;\n"
        "	ourTexCoord = texCoord;\n"
        "}\n";

static const char *glsl_drawtex_fragshader_src =
        "#version 330 core\n"
        "uniform usampler2D tex;\n"
        "uniform usampler1D palette_tex;\n"
        "uniform usampler2D mask;\n"
        "uniform bool interpolate = false;"
        "in vec3 ourColor;\n"
        "in vec2 ourTexCoord;\n"
        "out vec4 color;\n"
        "vec4 color_at_texcoord(vec2 v)"
        "{"
        "       uvec4 mask_value = texture(mask, v);"
        "       if(mask_value.x == uint(0)) { discard; }"
        "       uvec4 index = texture(tex, v);\n"
        "       vec4 c = texture(palette_tex, float(index) / 255.0);\n"
        "   	return vec4(c.rgb, 255) / 255.0;\n"
        "}"
        "vec4 shade_bilinear()\n"
        "{\n"
        "       float x = ourTexCoord.x * 320.0;"
        "       float y = ourTexCoord.y * 200.0;"
        "       float uv_x_step = 1.0 / 320.0;"
        "       float uv_y_step = 1.0 / 200.0;"
        "       float x0 = ceil(x - 1.0);"
        "       float y0 = ceil(y - 1.0);"
        "       vec4 color_x0_y0 = color_at_texcoord(ourTexCoord);"
        "       vec4 color_x1_y0 = color_at_texcoord(vec2(ourTexCoord.x + uv_x_step, ourTexCoord.y));"

        "       vec4 color_x0_y1 = color_at_texcoord(vec2(ourTexCoord.x, ourTexCoord.y + uv_y_step));"
        "       vec4 color_x1_y1 = color_at_texcoord(vec2(ourTexCoord.x + uv_x_step, ourTexCoord.y + uv_y_step));"
        "       vec4 color_y0 = mix(color_x0_y0, color_x1_y0, x - x0);"
        "       vec4 color_y1 = mix(color_x0_y1, color_x1_y1, x - x0);"
        "       return mix(color_y0, color_y1, y - y0);"
        "}\n"
        "void main()"
        "{"
        "       if(interpolate) {"
        "               color = shade_bilinear();"
        "       }"
        "       else {"
        "               color = color_at_texcoord(ourTexCoord);"
        "       }"
        "}";


#include "i_system.h"
#include "r_local.h"

#include "doomdef.h"
#include "doomdata.h"

#include "m_bbox.h"
#include "m_swap.h"

#include "v_video.h"
#include "r_opengl.h"
#include "r_geometry.h"
#include "rt_raytracing.cuh"
#include <GL/glew.h>
#include <stdlib.h>

// Each screen is [SCREENWIDTH*SCREENHEIGHT]; 
byte *screens[SCREEN_COUNT];

int dirtybox[4];

static SDL_Window *window;
static SDL_GLContext gl_context;
static GLuint VBO, VAO, EBO;
static GLuint fragment_shader, vertex_shader, shader_program;
static GLuint frame_texture;
static GLuint mask_texture;
static GLuint palette_texture;
GLubyte *pixels[SCREEN_COUNT];
GLubyte *mask;
static GLubyte current_palette[256 * 3];


// Now where did these came from?
byte gammatable[5][256] =
        {
                {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                                                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                                                                                                                 33, 34, 35, 36, 37, 38, 39,  40,  41,  42,  43,  44, 45,  46,  47,  48,
                                                                                                                                                                                                                          49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
                                                                                                                                                                                                                                                                                                          65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                                                                                                                                                                                                                                                                                                                                                                          81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                          97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255},

                {2,  4,  5,  7,  8,  10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31,
                                                                                                             32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49,  50,  51,  52,  54,  55,
                                                                                                                                                                                                      56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  69,  70,  71,  72,  73,  74,  75,  76,  77,
                                                                                                                                                                                                                                                                                                               78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,
                                                                                                                                                                                                                                                                                                                                                                                                                        99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 129,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        146, 147, 148, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        161, 162, 163, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 186, 187, 188, 189,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        190, 191, 192, 193, 194, 195, 196, 196, 197, 198, 199, 200, 201, 202, 203, 204,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        205, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 214, 215, 216, 217, 218,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        219, 220, 221, 222, 222, 223, 224, 225, 226, 227, 228, 229, 230, 230, 231, 232,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        233, 234, 235, 236, 237, 237, 238, 239, 240, 241, 242, 243, 244, 245, 245, 246,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        247, 248, 249, 250, 251, 252, 252, 253, 254, 255},

                {4,  7,  9,  11, 13, 15, 17, 19, 21, 22, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 40, 42,
                                                                                                             43, 45, 46, 47, 48, 50, 51, 52, 54, 55, 56, 57, 59, 60, 61, 62,  63,  65,  66,  67,  68, 69,
                                                                                                                                                                                                           70,  72,  73,  74,  75,  76,  77,  78,  79,  80,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
                                                                                                                                                                                                                                                                                                                         94,  95,  96,  97,  98,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                                                                                                                                                                                                                                                                                                                                                                                                   113, 114, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        129, 130, 131, 132, 133, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 153, 154, 155, 156, 157, 158, 159,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  160, 160, 161, 162, 163, 164, 165, 166, 166, 167, 168, 169, 170, 171, 172, 172, 173,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       174, 175, 176, 177, 178, 178, 179, 180, 181, 182, 183, 183, 184, 185, 186, 187, 188,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            188, 189, 190, 191, 192, 193, 193, 194, 195, 196, 197, 197, 198, 199, 200, 201, 201,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 202, 203, 204, 205, 206, 206, 207, 208, 209, 210, 210, 211, 212, 213, 213, 214, 215,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      216, 217, 217, 218, 219, 220, 221, 221, 222, 223, 224, 224, 225, 226, 227, 228, 228,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           229, 230, 231, 231, 232, 233, 234, 235, 235, 236, 237, 238, 238, 239, 240, 241, 241,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                242, 243, 244, 244, 245, 246, 247, 247, 248, 249, 250, 251, 251, 252, 253, 254, 254,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     255},

                {8,  12, 16, 19, 22, 24, 27, 29, 31, 34, 36, 38, 40, 41, 43, 45, 47, 49, 50, 52, 53, 55,
                                                                                                         57, 58, 60, 61, 63, 64, 65, 67, 68, 70, 71, 72, 74, 75, 76, 77, 79,  80,  81,  82,  84,  85,
                                                                                                                                                                                                      86,  87,  88,  90,  91,  92,  93,  94,  95,  96,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
                                                                                                                                                                                                                                                                                                          108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                                                                                                                                                                                                                                                                                                                                                                                               125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 135, 136, 137, 138, 139, 140,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    141, 142, 143, 143, 144, 145, 146, 147, 148, 149, 150, 150, 151, 152, 153, 154, 155,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         155, 156, 157, 158, 159, 160, 160, 161, 162, 163, 164, 165, 165, 166, 167, 168, 169,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              169, 170, 171, 172, 173, 173, 174, 175, 176, 176, 177, 178, 179, 180, 180, 181, 182,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   183, 183, 184, 185, 186, 186, 187, 188, 189, 189, 190, 191, 192, 192, 193, 194, 195,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        195, 196, 197, 197, 198, 199, 200, 200, 201, 202, 202, 203, 204, 205, 205, 206, 207,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             207, 208, 209, 210, 210, 211, 212, 212, 213, 214, 214, 215, 216, 216, 217, 218, 219,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  219, 220, 221, 221, 222, 223, 223, 224, 225, 225, 226, 227, 227, 228, 229, 229, 230,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       231, 231, 232, 233, 233, 234, 235, 235, 236, 237, 237, 238, 238, 239, 240, 240, 241,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            242, 242, 243, 244, 244, 245, 246, 246, 247, 247, 248, 249, 249, 250, 251, 251, 252,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 253, 253, 254, 254, 255},

                {16, 23, 28, 32, 36, 39, 42, 45, 48, 50, 53, 55, 57, 60, 62, 64, 66, 68, 69, 71, 73, 75, 76,
                                                                                                             78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 94, 96, 97, 98, 100, 101, 102, 103, 105, 106,
                                                                                                                                                                                                      107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                                                                                                                                                                                                                                                                                           125, 126, 128, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                                                                                                                                                                                                                                                                                                                                                                                142, 143, 143, 144, 145, 146, 147, 148, 149, 150, 150, 151, 152, 153, 154, 155, 155,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     156, 157, 158, 159, 159, 160, 161, 162, 163, 163, 164, 165, 166, 166, 167, 168, 169,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          169, 170, 171, 172, 172, 173, 174, 175, 175, 176, 177, 177, 178, 179, 180, 180, 181,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               182, 182, 183, 184, 184, 185, 186, 187, 187, 188, 189, 189, 190, 191, 191, 192, 193,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    193, 194, 195, 195, 196, 196, 197, 198, 198, 199, 200, 200, 201, 202, 202, 203, 203,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         204, 205, 205, 206, 207, 207, 208, 208, 209, 210, 210, 211, 211, 212, 213, 213, 214,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              214, 215, 216, 216, 217, 217, 218, 219, 219, 220, 220, 221, 221, 222, 223, 223, 224,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   224, 225, 225, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 232, 232, 233, 233,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        234, 234, 235, 235, 236, 236, 237, 237, 238, 239, 239, 240, 240, 241, 241, 242, 242,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 248, 248, 249, 249, 250, 250, 251,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  251, 252, 252, 253, 254, 254, 255, 255}
        };


int usegamma;

//
// V_MarkRect 
// 
void
V_MarkRect
        (int x,
         int y,
         int width,
         int height) {
    M_AddToBox(dirtybox, x, y);
    M_AddToBox(dirtybox, x + width - 1, y + height - 1);
}


//
// V_CopyRect 
// 
void
V_CopyRect
        (int srcx,
         int srcy,
         int srcscrn,
         int width,
         int height,
         int destx,
         int desty,
         int destscrn) {
    byte *src;
    byte *dest;
    byte *dest_mask;

#ifdef RANGECHECK
    if (srcx < 0
        || srcx + width > SCREENWIDTH
        || srcy < 0
        || srcy + height > SCREENHEIGHT
        || destx < 0 || destx + width > SCREENWIDTH
        || desty < 0
        || desty + height > SCREENHEIGHT
        || (unsigned) srcscrn > 4
        || (unsigned) destscrn > 4) {
        I_Error("Bad V_CopyRect");
    }
#endif
    V_MarkRect(destx, desty, width, height);

    src = pixels[srcscrn] + SCREENWIDTH * srcy + srcx;
    dest = pixels[destscrn] + SCREENWIDTH * desty + destx;
    dest_mask = mask + SCREENWIDTH * desty + destx;

    for (; height > 0; height--) {
        memcpy(dest, src, width);
        for(int i = 0; i < width; ++i) {
            dest_mask[i] = 0xFF;
        }
        src += SCREENWIDTH;
        dest += SCREENWIDTH;
        dest_mask += SCREENWIDTH;
    }
}

void
V_DrawPatch
        (int x,
         int y,
         int scrn,
         patch_t *patch) {

    column_t *column;
    byte *source;

    y -= SHORT(patch->topoffset);
    x -= SHORT(patch->leftoffset);
#ifdef RANGECHECK
    if (x < 0
        || x + SHORT(patch->width) > SCREENWIDTH
        || y < 0
        || y + SHORT(patch->height) > SCREENHEIGHT
        || (unsigned) scrn > 4) {
        fprintf(stderr, "Patch at %d,%d exceeds LFB\n", x, y);
        // No I_Error abort - what is up with TNT.WAD?
        fprintf(stderr, "V_DrawPatch: bad patch (ignored)\n");
        return;
    }
#endif
    for (int patch_column = 0; patch_column < patch->width; ++patch_column) {
        column = (column_t *) ((byte *) patch + LONG(patch->columnofs[patch_column]));

        while (column->topdelta != 0xff) {
            source = (byte *) column + 3;
            int screen_x = x + patch_column;

            for (int patch_y = 0; patch_y < column->length; ++patch_y) {
                int screen_y = y + column->topdelta + patch_y;
                int index = screen_y * SCREENWIDTH + screen_x;
                pixels[scrn][index] = *source++;
                if(scrn == 0) {
                    mask[index] = 0xFF;
                }

            }

            column = (column_t *) ((byte *) column + column->length
                                   + 4);
        }
    }
}

//
// V_DrawPatchFlipped 
// Masks a column based masked pic to the screen.
// Flips horizontally, e.g. to mirror face.
//
void
V_DrawPatchFlipped
        (int x,
         int y,
         int scrn,
         patch_t *patch) {

    int count;
    int col;
    column_t *column;
    byte *desttop;
    byte *dest;
    byte *source;
    int w;

    y -= SHORT(patch->topoffset);
    x -= SHORT(patch->leftoffset);
#ifdef RANGECHECK
    if (x < 0
        || x + SHORT(patch->width) > SCREENWIDTH
        || y < 0
        || y + SHORT(patch->height) > SCREENHEIGHT
        || (unsigned) scrn > 4) {
        fprintf(stderr, "Patch origin %d,%d exceeds LFB\n", x, y);
        I_Error("Bad V_DrawPatch in V_DrawPatchFlipped");
    }
#endif

    if (!scrn)
        V_MarkRect(x, y, SHORT(patch->width), SHORT(patch->height));

    col = 0;
    desttop = screens[scrn] + y * SCREENWIDTH + x;

    w = SHORT(patch->width);

    for (; col < w; x++, col++, desttop++) {
        column = (column_t *) ((byte *) patch + LONG(patch->columnofs[w - 1 - col]));

        // step through the posts in a column
        while (column->topdelta != 0xff) {
            source = (byte *) column + 3;
            dest = desttop + column->topdelta * SCREENWIDTH;
            count = column->length;

            while (count--) {
                *dest = *source++;
                dest += SCREENWIDTH;
            }
            column = (column_t *) ((byte *) column + column->length
                                   + 4);
        }
    }
}


//
// V_DrawPatchDirect
// Draws directly to the screen on the pc. 
//
void
V_DrawPatchDirect
        (int x,
         int y,
         int scrn,
         patch_t *patch) {
    V_DrawPatch(x, y, scrn, patch);

    /*
    int		count;
    int		col; 
    column_t*	column; 
    byte*	desttop;
    byte*	dest;
    byte*	source; 
    int		w; 
	 
    y -= SHORT(patch->topoffset); 
    x -= SHORT(patch->leftoffset); 

#ifdef RANGECHECK 
    if (x<0
	||x+SHORT(patch->width) >SCREENWIDTH
	|| y<0
	|| y+SHORT(patch->height)>SCREENHEIGHT 
	|| (unsigned)scrn>4)
    {
	I_Error ("Bad V_DrawPatchDirect");
    }
#endif 
 
    //	V_MarkRect (x, y, SHORT(patch->width), SHORT(patch->height)); 
    desttop = destscreen + y*SCREENWIDTH/4 + (x>>2); 
	 
    w = SHORT(patch->width); 
    for ( col = 0 ; col<w ; col++) 
    { 
	outp (SC_INDEX+1,1<<(x&3)); 
	column = (column_t *)((byte *)patch + LONG(patch->columnofs[col])); 
 
	// step through the posts in a column 
	 
	while (column->topdelta != 0xff ) 
	{ 
	    source = (byte *)column + 3; 
	    dest = desttop + column->topdelta*SCREENWIDTH/4; 
	    count = column->length; 
 
	    while (count--) 
	    { 
		*dest = *source++; 
		dest += SCREENWIDTH/4; 
	    } 
	    column = (column_t *)(  (byte *)column + column->length 
				    + 4 ); 
	} 
	if ( ((++x)&3) == 0 ) 
	    desttop++;	// go to next byte, not next plane 
    }*/
}


//
// V_DrawBlock
// Draw a linear block of pixels into the view buffer.
//
void
V_DrawBlock
        (int x,
         int y,
         int scrn,
         int width,
         int height,
         byte *src) {
    byte *dest;
    byte *dest_mask;

#ifdef RANGECHECK
    if (x < 0
        || x + width > SCREENWIDTH
        || y < 0
        || y + height > SCREENHEIGHT
        || (unsigned) scrn > 4) {
        I_Error("Bad V_DrawBlock");
    }
#endif

    V_MarkRect(x, y, width, height);
    dest = pixels[scrn] + y * SCREENWIDTH + x;
    dest_mask = mask + y * SCREENWIDTH + x;

    while (height--) {
        memcpy(dest, src, width);
        for(int i = 0; i < width; ++i) {
            dest_mask[i] = 0xFF;
        }
        src += width;
        dest += SCREENWIDTH;
        dest_mask += SCREENWIDTH;
    }
}


//
// V_GetBlock
// Gets a linear block of pixels from the view buffer.
//
void
V_GetBlock
        (int x,
         int y,
         int scrn,
         int width,
         int height,
         byte *dest) {
    byte *src;

#ifdef RANGECHECK
    if (x < 0
        || x + width > SCREENWIDTH
        || y < 0
        || y + height > SCREENHEIGHT
        || (unsigned) scrn > 4) {
        I_Error("Bad V_DrawBlock");
    }
#endif

    src = screens[scrn] + y * SCREENWIDTH + x;

    while (height--) {
        memcpy(dest, src, width);
        src += SCREENWIDTH;
        dest += width;
    }
}

// QUAD GEOMETRY
static GLfloat vertices[] = {
        // Positions          // Colors           // Texture Coords
        1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,  // Top Right
        1.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,  // Bottom Right
        -1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,  // Bottom Left
        -1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f // Top Left
};
// you can also put positions, colors and coordinates in seperate VBO's
static GLuint indices[] = {  // Note that we start from 0!
        0, 1, 3,  // First Triangle
        1, 2, 3   // Second Triangle
};

static void init_sdl_window(boolean fullscreen) {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    int width = SCREENWIDTH;
    int height = SCREENHEIGHT;

    int flags = SDL_WINDOW_OPENGL;

    if (fullscreen == true) {
        flags |= SDL_WINDOW_FULLSCREEN;
        int display_count = SDL_GetNumVideoDisplays();

        SDL_DisplayMode mode;
        SDL_GetCurrentDisplayMode(0, &mode);
        width = mode.w;
        height = mode.h;
    }

    window = SDL_CreateWindow("DOOM!",
                              SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED,
                              width, height,
                              flags);

    gl_context = SDL_GL_CreateContext(window);
    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE; // need this to enforce core profile
    GLenum err = glewInit();
    glGetError();
    if (err != GLEW_OK) {
        I_Error("glewInit failed: %s", glewGetErrorString(err));
    }

    glViewport(0, 0, width, height);

    R_CheckForGlErrors();
}

GLuint link_program(GLuint vertex_shader, GLuint fragment_shader) {
    // create empty program
    GLuint program = glCreateProgram();
    // try to attach all shaders
    GLuint shaders[2] = {vertex_shader - fragment_shader};
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    // try to link program
    glLinkProgram(program);
    GLint is_linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &is_linked); // check if program linked
    if (!is_linked) {

        GLint infologLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, (GLint *) &infologLength);
        char *infoLog = (char *) malloc(infologLength);
        glGetProgramInfoLog(program, infologLength, NULL, infoLog); // will include terminate char
        glDeleteProgram(program);
        I_Error("Program compilation error: %s", infoLog);
        free(infoLog);
    }

    return program;
}

GLuint compile_shader(GLenum shader_type, const char *source) {
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    // check if shader compiled
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {

        GLint infologLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);
        char *infoLog = (char *) malloc(infologLength);
        glGetShaderInfoLog(shader, infologLength, NULL, infoLog); // will include terminate char
        glDeleteShader(shader);
        I_Error("Shader compilation error: %s", infoLog);
        free(infoLog);
    }

    return shader;
}

GLuint create_frame_texture(int width, int height, GLubyte *buffer) {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, 320, 200, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, buffer);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, width, height, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);

    R_CheckForGlErrors();

    return texture;
}

GLuint create_palette_texture() {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_1D, texture);
    R_CheckForGlErrors();

    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    R_CheckForGlErrors();

    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB8UI, 256, 0, GL_RGB_INTEGER, GL_UNSIGNED_BYTE, current_palette);

    R_CheckForGlErrors();

    return texture;
}

static void init_gl_buffers() {
    // Generate buffers
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Buffer setup
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid *) 0);
    glEnableVertexAttribArray(0);
    // Color attribute (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid *) (3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid *) (6 * sizeof(GLfloat)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound
    // vertex buffer object so afterwards we can safely unbind
    glBindVertexArray(0);

    R_CheckForGlErrors();
}

void init_gl_textures() {
    frame_texture = create_frame_texture(SCREENWIDTH, SCREENHEIGHT, pixels[0]);
    mask_texture = create_frame_texture(SCREENWIDTH, SCREENHEIGHT, mask);
    palette_texture = create_palette_texture();
}

void init_gl_shaders() {
    vertex_shader = compile_shader(GL_VERTEX_SHADER, glsl_drawtex_vertshader_src);
    fragment_shader = compile_shader(GL_FRAGMENT_SHADER, glsl_drawtex_fragshader_src);
    shader_program = link_program(vertex_shader, fragment_shader);
}

//
// V_Init
// 
void V_Init(void) {
    int i;
    byte *base;

    // stick these in low dos memory on PCs

    base = I_AllocLow(SCREENWIDTH * SCREENHEIGHT * SCREEN_COUNT);

    for (i = 0; i < SCREEN_COUNT; i++) {
        screens[i] = base + i * SCREENWIDTH * SCREENHEIGHT;
        pixels[i] = (GLubyte *) malloc(SCREENWIDTH * SCREENHEIGHT);
        memset(pixels[i], 0x00, SCREENWIDTH * SCREENHEIGHT);
    }

    mask = (GLubyte *) malloc(SCREENWIDTH * SCREENHEIGHT);
    memset(mask, 0x00, SCREENWIDTH * SCREENHEIGHT);

    init_sdl_window(false);
    init_gl_buffers();
    init_gl_textures();
    init_gl_shaders();

    glDepthMask(false);

    R_CheckForGlErrors();
}

void V_UpdatePalette(byte *palette) {
    memcpy(current_palette, palette, 256 * 3);
    glBindTexture(GL_TEXTURE_1D, palette_texture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB8UI, 256, 0, GL_RGB_INTEGER, GL_UNSIGNED_BYTE, current_palette);

    R_CheckForGlErrors();
}

void update_frame_texture(void) {
    /*glTextureSubImage2D(frame_texture,
                        0,
                        0,
                        0,
                        320,
                        200,
                        GL_RGBA_INTEGER_EXT,
                        GL_UNSIGNED_BYTE,
                        pixels);*/

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, frame_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, 320, 200, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, pixels[0]);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mask_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, 320, 200, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, mask);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, 320, 200, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, pixels);

    R_CheckForGlErrors();
}

void clear_mask_texture() {
    auto mask_value = RT_IsEnabled() ? 0x00 : 0xFF;
    memset(mask, mask_value, SCREENWIDTH*SCREENHEIGHT);
}

void V_Render(void) {
    update_frame_texture();
    //glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    //glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shader_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, frame_texture);
    glUniform1i(glGetUniformLocation(shader_program, "tex"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D, palette_texture);
    glUniform1i(glGetUniformLocation(shader_program, "palette_tex"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mask_texture);
    glUniform1i(glGetUniformLocation(shader_program, "mask"), 2);

    //R_DrawGeometry();
    glBindVertexArray(VAO); // binding VAO automatically binds EBO
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0); // unbind VAO

    R_CheckForGlErrors();
    clear_mask_texture();
}

void V_Swap(void) {
    SDL_GL_SwapWindow(window);
}

void V_ToggleFullScreen() {
    Uint32 fullscreen_flag = SDL_WINDOW_FULLSCREEN;
    unsigned int is_fullscreen = SDL_GetWindowFlags(window) & fullscreen_flag;

    SDL_DestroyWindow(window);
    init_sdl_window(is_fullscreen == 0);

    // TODO Cleanup buffers before recreating them
    init_gl_buffers();
    init_gl_textures();
    init_gl_shaders();

    R_CheckForGlErrors();
}