#version 460

layout(location = 0) in mat2x3 vertex_attributes;
layout(location = 0) out vec3 vertex_tex_coords;

void main() {
	vec3 vertex_position = vertex_attributes[0];
    gl_Position = vec4(vertex_position, 1.0);
	vertex_tex_coords = vertex_attributes[1]; 
}