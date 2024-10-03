#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "ao_common.glsl"

layout(location = 1) rayPayloadInEXT AOPayload incoming_payload;

void main() {
    incoming_payload.t = 1.0 / (gl_RayTmaxEXT * gl_RayTmaxEXT);
}