#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"

struct Voxel {
  vec4 position;
};

layout(location = 0) rayPayloadInEXT RayPayload incoming_payload;

layout(set = 0, binding = 2, scalar) uniform _vec3 { vec3 colors[256]; } colors_buffer;
layout(set = 0, binding = 3, scalar) buffer voxels { Voxel allVoxels[]; };

void main() {
    vec3 normals[6] = vec3[](
        vec3(0., -1., 0.),
        vec3(1., 0., 0.),
        vec3(0., 0., -1.),
        vec3(-1., 0., 0.),
        vec3(0., 0., 1.),
        vec3(0., 1., 0.)
    );
    const Voxel voxel = allVoxels[gl_PrimitiveID];

    const vec3 hit_point = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_RayTmaxEXT;
    const vec3 normal = normals[gl_HitKindEXT];

    incoming_payload.color = colors_buffer.colors[gl_InstanceCustomIndexEXT];
    incoming_payload.origin = hit_point;
    incoming_payload.direction = reflect(gl_WorldRayDirectionEXT, normal);
    incoming_payload.t = 1.0;
}