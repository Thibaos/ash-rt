#![allow(unused)]

use ash::{
    vk::{self, AabbPositionsKHR},
    Device,
};

use crate::{
    uniform_types::VoxelInfos,
    utils::{get_buffer_device_address, BufferResource},
};

pub fn open_file(path: &str) -> dot_vox::DotVoxData {
    let vox_data = dot_vox::load(path).unwrap();

    #[cfg(debug_assertions)]

    println!("Palette has {} colors", vox_data.palette.len());

    // TODO scene support
    #[cfg(debug_assertions)]
    {
        for scene in vox_data.scenes.iter() {
            match scene {
                dot_vox::SceneNode::Group {
                    attributes,
                    children,
                } => {
                    println!(
                        "group: {} attributes, {} models",
                        attributes.len(),
                        children.len()
                    );
                }
                dot_vox::SceneNode::Shape { attributes, models } => {
                    println!(
                        "shape: {} attributes, {} models",
                        attributes.len(),
                        models.len()
                    );
                }
                dot_vox::SceneNode::Transform {
                    attributes,
                    frames,
                    child,
                    layer_id,
                } => {
                    println!(
                        "transform: {} attributes, {} frames, {} child, {} layer_id",
                        attributes.len(),
                        frames.len(),
                        child,
                        layer_id
                    );
                }
            }
        }
    }

    vox_data
}

pub fn vox_to_tlas(
    as_device_handle: u64,
    input_voxels: Vec<dot_vox::Voxel>,
) -> (Vec<vk::AccelerationStructureInstanceKHR>, Vec<VoxelInfos>) {
    let mut instances = Vec::<vk::AccelerationStructureInstanceKHR>::new();
    let mut positions = Vec::<VoxelInfos>::new();

    for vox in input_voxels {
        let x = f32::from(vox.x);
        let y = f32::from(vox.z);
        let z = f32::from(vox.y);
        let transform: [f32; 12] = [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z];

        // TODO palette support
        let instance = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR { matrix: transform },
            instance_custom_index_and_mask: vk::Packed24_8::new(vox.i.into(), 0xff),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: as_device_handle,
            },
        };

        instances.push(instance);
        positions.push(VoxelInfos {
            position: glm::Vec3::new(x, y, z),
            palette_index: 0,
        });
    }

    println!("Loaded {} voxels", instances.len());

    (instances, positions)
}

pub fn vox_to_blas<'a>(
    input_voxels: &Vec<dot_vox::Voxel>,
    device: &Device,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
) -> vk::AccelerationStructureGeometryKHR<'a> {
    let aabb_buffer = {
        let mut corners = Vec::<AabbPositionsKHR>::new();

        for voxel in input_voxels {
            let x = f32::from(voxel.x);
            let y = f32::from(voxel.z);
            let z = f32::from(voxel.y);

            corners.push(AabbPositionsKHR {
                min_x: x - 0.5,
                min_y: y - 0.5,
                min_z: z - 0.5,
                max_x: x + 0.5,
                max_y: y + 0.5,
                max_z: z + 0.5,
            })
        }

        let aabb_stride = std::mem::size_of::<vk::AabbPositionsKHR>();
        let buffer_size = (aabb_stride * corners.len()) as vk::DeviceSize;

        let mut aabb_buffer = BufferResource::new(
            buffer_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            device,
            device_memory_properties,
        );

        aabb_buffer.store(&corners, device);

        aabb_buffer
    };

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::AABBS)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default().data(
                vk::DeviceOrHostAddressConstKHR {
                    device_address: unsafe {
                        get_buffer_device_address(device, aabb_buffer.buffer)
                    },
                },
            ),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    geometry
}

pub fn vox_to_geometries<'a>(
    input_voxels: &Vec<dot_vox::Voxel>,
    device: &Device,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
) -> (BufferResource, vk::AccelerationStructureGeometryKHR<'a>) {
    let voxel_count = input_voxels.len();

    let aabb_buffer = {
        let mut corners = Vec::<AabbPositionsKHR>::new();

        for voxel in input_voxels {
            let x = f32::from(voxel.x);
            let y = f32::from(voxel.z);
            let z = f32::from(voxel.y);

            corners.push(AabbPositionsKHR {
                min_x: x - 0.5,
                min_y: y - 0.5,
                min_z: z - 0.5,
                max_x: x + 0.5,
                max_y: y + 0.5,
                max_z: z + 0.5,
            })
        }

        let aabb_stride = std::mem::size_of::<vk::AabbPositionsKHR>();
        let buffer_size = (aabb_stride * corners.len()) as vk::DeviceSize;

        let mut aabb_buffer = BufferResource::new(
            buffer_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            device,
            device_memory_properties,
        );

        aabb_buffer.store(&corners, device);

        aabb_buffer
    };

    // let host_data = unsafe { std::mem::transmute(0u8) };

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::AABBS)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .index_type(vk::IndexType::UINT32)
                .max_vertex(3),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    // aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default().data(
    //     vk::DeviceOrHostAddressConstKHR {
    //         device_address: unsafe {
    //             get_buffer_device_address(device, aabb_buffer.buffer)
    //         },
    //     },
    // ),

    (aabb_buffer, geometry)
}

pub fn get_palette(data: &dot_vox::DotVoxData) -> [glm::Vec3; 256] {
    let mut array = [glm::Vec3::zeros(); 256];
    for i in 0..256 {
        let color = data.palette.get(i).unwrap();
        array[i] = glm::Vec3::new(
            f32::from(color.r) / 255.0,
            f32::from(color.g) / 255.0,
            f32::from(color.b) / 255.0,
        )
    }

    array
}
