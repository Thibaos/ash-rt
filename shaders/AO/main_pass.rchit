#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "ao_common.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;

struct Voxel {
  vec3 position;
  uint palette_index;
};

struct Vertex {
    vec3 position;
};

struct Index {
    uint index;
};

layout(set = 0, binding = 2, scalar) uniform _vec3 { vec3 palette[256]; } palette_buffer;
layout(set = 0, binding = 3, scalar) buffer _Voxels { Voxel voxels[]; };
layout(set = 0, binding = 4, scalar) buffer Vertices { Vertex v[]; };
layout(set = 0, binding = 5, scalar) buffer Indices { uint i[]; };

void main() {
    const Voxel voxel = voxels[gl_InstanceID];

    // Indices of the triangle
    uint base_index = i[gl_PrimitiveID * 3];

    // Vertex of the triangle
    Vertex v0 = v[base_index];
    Vertex v1 = v[base_index + 1];
    Vertex v2 = v[base_index + 2];

    const vec3 normals[6] = vec3[](
        vec3(0., 0., 1.), // +z
        vec3(1., 0., 0.), // +x
        vec3(0., 0., -1.), // -z
        vec3(-1., 0., 0.), // -x
        vec3(0., -1., 0.), // -y
        vec3(0., 1., 0.) // +y
    );

    uint face_index = (gl_PrimitiveID / 2);
    vec3 normal = normals[face_index];

    incoming_payload.normal = normal;
    incoming_payload.color = palette_buffer.palette[voxel.palette_index];
    incoming_payload.t = gl_RayTmaxEXT;
}