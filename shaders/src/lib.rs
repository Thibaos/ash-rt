#![no_std]

use spirv_std::{
    arch::report_intersection,
    glam::{uvec2, vec2, vec3, vec4, Mat4, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles},
    image::Image,
    ray_tracing::{AccelerationStructure, RayFlags},
    spirv,
};

#[repr(C)]
pub struct Aabb {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
}

// Uniform buffer set at each frame
#[repr(C)]
pub struct GlobalUniforms {
    pub origin: Vec3,
    pub direction: Vec3,
    pub view_proj: Mat4,    // Camera view * projection
    pub view_inverse: Mat4, // Camera inverse view matrix
    pub proj_inverse: Mat4, // Camera inverse projection matrix
}

#[allow(unused)]
// Ray-AABB intersection
fn hit_aabb(aabb: &Aabb, origin: Vec3, direction: Vec3) -> f32 {
    let inv_dir = 1.0 / direction;
    let minimum = Vec3 {
        x: aabb.min_x,
        y: aabb.min_y,
        z: aabb.min_z,
    };
    let maximum = Vec3 {
        x: aabb.max_x,
        y: aabb.max_y,
        z: aabb.max_z,
    };
    let tbot = inv_dir * (minimum - origin);
    let ttop = inv_dir * (maximum - origin);
    let tmin = Vec3::min(ttop, tbot);
    let tmax = Vec3::max(ttop, tbot);
    let t0 = f32::max(tmin.x, f32::max(tmin.y, tmin.z));
    let t1 = f32::min(tmax.x, f32::min(tmax.y, tmax.z));

    if t1 > f32::max(t0, 0.0) {
        t0
    } else {
        -1.0
    }
}

#[spirv(fragment)]
pub fn main_fs(output: &mut Vec4, color: Vec3) {
    *output = color.extend(1.0);
}

#[spirv(vertex)]
pub fn main_vs(
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(position, invariant)] out_pos: &mut Vec4,
    color: &mut Vec3,
) {
    *out_pos = vec4(
        (vert_id - 1) as f32,
        ((vert_id & 1) * 2 - 1) as f32,
        0.0,
        1.0,
    );

    *color = [
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0),
    ][vert_id as usize];
}

#[spirv(miss)]
pub fn main_miss(#[spirv(incoming_ray_payload)] _out: &mut Vec3) {}

#[spirv(closest_hit)]
pub fn main_closest_hit(
    #[spirv(instance_custom_index)] index: usize,
    #[spirv(uniform, descriptor_set = 0, binding = 2)] color_buffer: &[Vec3; 3],
    #[spirv(incoming_ray_payload)] out: &mut Vec3,
) {
    *out = color_buffer[index];
}

#[spirv(intersection)]
pub fn main_intersection() {
    unsafe {
        report_intersection(1.0, 0);
    }
}

#[spirv(ray_generation)]
pub fn main_ray_generation(
    #[spirv(launch_id)] launch_id: UVec3,
    #[spirv(launch_size)] launch_size: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] top_level_as: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1)] image: &Image!(2D, format = rgba8, sampled = false),
    #[spirv(uniform, descriptor_set = 1, binding = 0)] globals: &GlobalUniforms,
    #[spirv(ray_payload)] payload: &mut Vec3,
) {
    let pixel_center = vec2(launch_id.x as f32, launch_id.y as f32) + vec2(0.5, 0.5);
    let in_uv = pixel_center / vec2(launch_size.x as f32, launch_size.y as f32);

    let d = in_uv * 2.0 - Vec2::ONE;

    let origin = (globals.view_inverse * vec4(0.0, 0.0, 0.0, 1.0)).xyz();
    let target = (globals.proj_inverse * vec4(d.x, -d.y, 1.0, 1.0)).xyz();
    let direction = (globals.view_inverse * target.normalize().extend(0.0)).xyz();

    let cull_mask = 0xff;
    let tmin = 0.001;
    let tmax = 1000.0;

    *payload = Vec3::ZERO;

    unsafe {
        top_level_as.trace_ray(
            RayFlags::OPAQUE,
            cull_mask,
            0,
            0,
            0,
            origin,
            tmin,
            direction,
            tmax,
            payload,
        );
    }

    unsafe {
        image.write(uvec2(launch_id.x, launch_id.y), payload.extend(1.0));
    }
}
