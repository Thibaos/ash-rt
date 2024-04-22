#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference2 : require

layout(location = 0) rayPayloadInEXT vec3 incoming_payload;

layout(set = 0, binding = 2, scalar) uniform _vec3 { vec3 colors[3];} colors_buffer;

void main() {
    incoming_payload = colors_buffer.colors[gl_InstanceCustomIndexEXT];
}