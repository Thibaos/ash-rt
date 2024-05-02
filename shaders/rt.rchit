#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference2 : require

layout(location = 0) rayPayloadInEXT vec3 incoming_payload;

layout(set = 0, binding = 2, scalar) uniform _vec3 { vec3 colors[3]; } colors_buffer;

void main() {
    vec3 normals[6] = vec3[](vec3(0.0, -1.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0), vec3(-1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0));
    // incoming_payload = colors_buffer.colors[gl_InstanceCustomIndexEXT];
    incoming_payload = vec3(0.1);
    if (gl_HitKindEXT < 6) {
        incoming_payload = colors_buffer.colors[gl_InstanceCustomIndexEXT] * dot(normals[gl_HitKindEXT], gl_WorldRayDirectionEXT.xyz);
    }
}