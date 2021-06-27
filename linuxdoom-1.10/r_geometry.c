#include "r_geometry.h"
#include "r_state.h"
#include <GL/glew.h>
#include <stdlib.h>

static GLuint vao;
static GLuint vbo;
static GLuint ebo;
static GLuint index_count;

static float fixed16_to_float(fixed_t val) {
    return (float)val / 0x00010000;
}

void R_BuildGeometry(void) {
    const int vertex_components = 8; // 3 for position, 3 for vertex color, 2 for texture coords

    int max_vertices = numlines * 8; // Two quads per side, 4 vertices per quad. At most.
    int max_indices = numlines * 12; // Two quads per side, 6 indices per quad. At most.
    int vertices_len = sizeof(GLfloat) * vertex_components * max_vertices;
    GLfloat *vertices = malloc(vertices_len);
    GLfloat *vertices_write_ptr = vertices;

    int indices_len = sizeof(GLuint) * max_indices;
    GLuint *indices = malloc(indices_len);
    GLuint *indices_write_ptr = indices;

    index_count = 0;
    GLuint vertex_count = 0;
    for(int i = 0; i < numlines; ++i) {
        line_t* line = &lines[i];
        for(int j = 0; j < 2; ++j) {

            int side_number = line->sidenum[j];
            if(side_number == -1) {
                continue;
            }

            side_t *side = &sides[side_number];

            // side->sector->floorheight

            vertex_t *v1 = line->v1;
            vertex_t *v2 = line->v2;

            *vertices_write_ptr++ = fixed16_to_float(v1->x);
            *vertices_write_ptr++ = fixed16_to_float(v1->y);
            *vertices_write_ptr++ = fixed16_to_float(side->sector->floorheight);
            *vertices_write_ptr++ = 1.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;

            *vertices_write_ptr++ = fixed16_to_float(v1->x);
            *vertices_write_ptr++ = fixed16_to_float(v1->y);
            *vertices_write_ptr++ = fixed16_to_float(side->sector->ceilingheight);
            *vertices_write_ptr++ = 1.0f;

            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;

            *vertices_write_ptr++ = fixed16_to_float(v2->x);
            *vertices_write_ptr++ = fixed16_to_float(v2->y);
            *vertices_write_ptr++ = fixed16_to_float(side->sector->ceilingheight);
            *vertices_write_ptr++ = 1.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;

            *vertices_write_ptr++ = fixed16_to_float(v2->x);
            *vertices_write_ptr++ = fixed16_to_float(v2->y);
            *vertices_write_ptr++ = fixed16_to_float(side->sector->floorheight);
            *vertices_write_ptr++ = 1.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;
            *vertices_write_ptr++ = 0.0f;

            // 0, 1, 3,  // First Triangle
            // 1, 2, 3   // Second Triangle
            *indices_write_ptr++ = vertex_count + 0;
            *indices_write_ptr++ = vertex_count + 1;
            *indices_write_ptr++ = vertex_count + 3;
            *indices_write_ptr++ = vertex_count + 1;
            *indices_write_ptr++ = vertex_count + 2;
            *indices_write_ptr++ = vertex_count + 3;
            index_count += 6;

            vertex_count += 4;
        }

    }

    // Generate buffers
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    // Buffer setup
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices_len, vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_len, indices, GL_STATIC_DRAW);

    // Position attribute (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid *) 0);
    glEnableVertexAttribArray(0);
    // Color attribute (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid *) (3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    // Texture attribute (2 floats)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid *) (6 * sizeof(GLfloat)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound
    // vertex buffer object so afterwards we can safely unbind
    glBindVertexArray(0);
}

void R_DrawGeometry(void) {
    if(index_count == 0) {
        return;
    }
    glBindVertexArray(vao); // binding VAO automatically binds EBO
    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0); // unbind VAO
}
