use bevy_transform::components::Transform;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Debug, Copy, Pod, Zeroable)]
pub struct GlobalUniforms {
    pub view_inverse: bevy_math::Mat4, // Camera inverse view matrix
    pub proj_inverse: glm::Mat4,       // Camera inverse projection matrix
}

#[repr(C)]
#[derive(Clone, Debug, Copy, Pod, Zeroable)]
pub struct VoxelInfos {
    pub position: glm::Vec3,
    pub palette_index: u32,
}

pub struct CameraTransform {
    pub transform: Transform,
}
