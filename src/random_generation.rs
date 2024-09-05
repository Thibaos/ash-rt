use ash::vk;
use rand::Rng;

use crate::uniform_types::VoxelInfos;

#[allow(unused)]
pub fn create_cube_instances(
    accel_handle: u64,
    n: u64,
    p: f32,
) -> (Vec<vk::AccelerationStructureInstanceKHR>, Vec<VoxelInfos>) {
    let mut instances = Vec::<vk::AccelerationStructureInstanceKHR>::new();
    let mut positions = Vec::<VoxelInfos>::new();

    let mut rng = rand::thread_rng();

    #[cfg(debug_assertions)]
    println!("Starting voxel cube build...");

    for i in 0..n * n * n {
        let x = i / (n * n);
        let y = (i / n) % n;
        let z = i % n;

        if x % 16 == 0 && y == 0 && z == 0 {
            #[cfg(debug_assertions)]
            println!("{:.1}%", (i as f32 / (n * n * n) as f32) * 100.0);
        }
        //
        // x % 3 == 0 && y % 3 == 0 && z % 3 == 0
        if rng.gen::<f32>() < p {
            let transform: [f32; 12] = [
                1.0, 0.0, 0.0, x as f32, 0.0, 1.0, 0.0, y as f32, 0.0, 0.0, 1.0, z as f32,
            ];

            let custom_index = rng.gen_range(0..3);

            let instance = vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: transform },
                instance_custom_index_and_mask: vk::Packed24_8::new(custom_index, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: accel_handle,
                },
            };

            instances.push(instance);
            positions.push(VoxelInfos {
                position: glm::Vec3::new(x as f32, y as f32, z as f32),
                palette_index: 0,
            });
        }
    }

    #[cfg(debug_assertions)]
    println!("Voxel cube built, {} voxels.", instances.len());

    (instances, positions)
}
