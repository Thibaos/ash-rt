#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "ao_common.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;

void main() {
    incoming_payload.color = vec3(0.0);
    incoming_payload.normal = vec3(0.0);
    incoming_payload.t = 0.0;
}