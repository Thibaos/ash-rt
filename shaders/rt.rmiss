#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"
#include "noise.glsl"

layout(location = 0) rayPayloadInEXT RayPayload incoming_payload;

void main() {
    vec3 base_direction = gl_WorldRayDirectionEXT;

    float n = snoise(base_direction);
    n += 0.5 * snoise(base_direction * 2.0);
    n += 0.25 * snoise(base_direction * 4.0);
    n += 0.125 * snoise(base_direction * 8.0);
    n += 0.0625 * snoise(base_direction * 16.0);
    n += 0.03125 * snoise(base_direction * 32.0);

    incoming_payload.color = vec3(n);
}