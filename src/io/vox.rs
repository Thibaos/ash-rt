use ash::vk;

use crate::uniform_types::VoxelPosition;

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
) -> (
    Vec<vk::AccelerationStructureInstanceKHR>,
    Vec<VoxelPosition>,
) {
    let mut instances = Vec::<vk::AccelerationStructureInstanceKHR>::new();
    let mut positions = Vec::<VoxelPosition>::new();

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
        positions.push(VoxelPosition {
            position: glm::Vec3::new(x, y, z),
            _pad: 0.0,
        });
    }

    println!("Loaded {} voxels", instances.len());

    (instances, positions)
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
