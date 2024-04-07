use ash::{
    extensions::khr::AccelerationStructure,
    vk::{
        self, AccelerationStructureGeometryKHR, AccelerationStructureKHR, CommandPool, Packed24_8,
        PhysicalDeviceMemoryProperties, Queue,
    },
    Device,
};

use crate::{base::ExampleBase, get_buffer_device_address, BufferResource};

pub fn create_geometry(
    base: &ExampleBase,
    device_memory_properties: PhysicalDeviceMemoryProperties,
) -> (AccelerationStructureGeometryKHR, BufferResource) {
    let aabb_buffer = {
        let corners = [vk::AabbPositionsKHR {
            min_x: 0.0,
            min_y: 0.0,
            min_z: 0.0,
            max_x: 1.0,
            max_y: 1.0,
            max_z: 1.0,
        }];

        let aabb_stride = std::mem::size_of::<vk::AabbPositionsKHR>();
        let buffer_size = (aabb_stride * corners.len()) as vk::DeviceSize;

        let mut aabb_buffer = BufferResource::new(
            buffer_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &base.device,
            device_memory_properties,
        );

        aabb_buffer.store(&corners, &base.device);

        aabb_buffer
    };

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::AABBS)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::builder()
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: unsafe {
                        get_buffer_device_address(&base.device, aabb_buffer.buffer)
                    },
                })
                .build(),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .build();

    (geometry, aabb_buffer)
}

pub fn create_bottom_as(
    acceleration_structure: &AccelerationStructure,
    geometry: AccelerationStructureGeometryKHR,
    device: &Device,
    device_memory_properties: PhysicalDeviceMemoryProperties,
    command_pool: CommandPool,
    graphics_queue: Queue,
) -> (AccelerationStructureKHR, BufferResource) {
    let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .primitive_count(1)
        .primitive_offset(0)
        .transform_offset(0)
        .build();

    let geometries = [geometry];

    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(&geometries)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .build();

    let size_info = unsafe {
        acceleration_structure.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[1],
        )
    };

    let bottom_as_buffer = BufferResource::new(
        size_info.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        &device,
        device_memory_properties,
    );

    let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .ty(build_info.ty)
        .size(size_info.acceleration_structure_size)
        .buffer(bottom_as_buffer.buffer)
        .offset(0)
        .build();

    let bottom_as =
        unsafe { acceleration_structure.create_acceleration_structure(&as_create_info, None) }
            .unwrap();

    build_info.dst_acceleration_structure = bottom_as;

    let scratch_buffer = BufferResource::new(
        size_info.build_scratch_size,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        &device,
        device_memory_properties,
    );

    build_info.scratch_data = vk::DeviceOrHostAddressKHR {
        device_address: unsafe { get_buffer_device_address(&device, scratch_buffer.buffer) },
    };

    let build_command_buffer = {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .build();

        let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap();
        command_buffers[0]
    };

    unsafe {
        device
            .begin_command_buffer(
                build_command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build(),
            )
            .unwrap();

        acceleration_structure.cmd_build_acceleration_structures(
            build_command_buffer,
            &[build_info],
            &[&[build_range_info]],
        );
        device.end_command_buffer(build_command_buffer).unwrap();
        device
            .queue_submit(
                graphics_queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[build_command_buffer])
                    .build()],
                vk::Fence::null(),
            )
            .expect("queue submit failed.");

        device.queue_wait_idle(graphics_queue).unwrap();
        device.free_command_buffers(command_pool, &[build_command_buffer]);
        scratch_buffer.destroy(&device);
    }

    (bottom_as, bottom_as_buffer)
}

pub fn create_instances(
    acceleration_structure: &AccelerationStructure,
    bottom_as: AccelerationStructureKHR,
    device: &Device,
    device_memory_properties: PhysicalDeviceMemoryProperties,
) -> (usize, BufferResource) {
    let accel_handle = {
        let as_addr_info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
            .acceleration_structure(bottom_as)
            .build();
        unsafe { acceleration_structure.get_acceleration_structure_device_address(&as_addr_info) }
    };

    let transform_0: [f32; 12] = [1.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let transform_1: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0];
    let transform_2: [f32; 12] = [1.0, 0.0, 0.0, 1.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];

    let instances = vec![
        vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: transform_0,
            },
            instance_custom_index_and_mask: Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: accel_handle,
            },
        },
        vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: transform_1,
            },
            instance_custom_index_and_mask: Packed24_8::new(1, 0xff),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: accel_handle,
            },
        },
        vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: transform_2,
            },
            instance_custom_index_and_mask: Packed24_8::new(2, 0xff),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: accel_handle,
            },
        },
    ];

    let instance_buffer_size =
        std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * instances.len();

    let mut instance_buffer = BufferResource::new(
        instance_buffer_size as vk::DeviceSize,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        &device,
        device_memory_properties,
    );

    instance_buffer.store(&instances, &device);

    (instances.len(), instance_buffer)
}

pub fn create_top_as(
    acceleration_structure: &AccelerationStructure,
    instance_count: usize,
    instance_buffer: &BufferResource,
    device: &Device,
    device_memory_properties: PhysicalDeviceMemoryProperties,
    command_pool: CommandPool,
    graphics_queue: Queue,
) -> (AccelerationStructureKHR, BufferResource) {
    let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .first_vertex(0)
        .primitive_count(instance_count as u32)
        .primitive_offset(0)
        .transform_offset(0)
        .build();

    let build_command_buffer = {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .build();

        let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap();
        command_buffers[0]
    };

    unsafe {
        device
            .begin_command_buffer(
                build_command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build(),
            )
            .unwrap();
        let memory_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .build();
        device.cmd_pipeline_barrier(
            build_command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[memory_barrier],
            &[],
            &[],
        );
    }

    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::builder()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: unsafe { get_buffer_device_address(&device, instance_buffer.buffer) },
        })
        .build();

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances })
        .build();

    let geometries = [geometry];

    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(&geometries)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .build();

    let size_info = unsafe {
        acceleration_structure.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[build_range_info.primitive_count],
        )
    };

    let top_as_buffer = BufferResource::new(
        size_info.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        &device,
        device_memory_properties,
    );

    let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .ty(build_info.ty)
        .size(size_info.acceleration_structure_size)
        .buffer(top_as_buffer.buffer)
        .offset(0)
        .build();

    let top_as =
        unsafe { acceleration_structure.create_acceleration_structure(&as_create_info, None) }
            .unwrap();

    build_info.dst_acceleration_structure = top_as;

    let scratch_buffer = BufferResource::new(
        size_info.build_scratch_size,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        &device,
        device_memory_properties,
    );

    build_info.scratch_data = vk::DeviceOrHostAddressKHR {
        device_address: unsafe { get_buffer_device_address(&device, scratch_buffer.buffer) },
    };

    unsafe {
        acceleration_structure.cmd_build_acceleration_structures(
            build_command_buffer,
            &[build_info],
            &[&[build_range_info]],
        );
        device.end_command_buffer(build_command_buffer).unwrap();
        device
            .queue_submit(
                graphics_queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[build_command_buffer])
                    .build()],
                vk::Fence::null(),
            )
            .expect("queue submit failed.");

        device.queue_wait_idle(graphics_queue).unwrap();
        device.free_command_buffers(command_pool, &[build_command_buffer]);
        scratch_buffer.destroy(&device);
    }

    (top_as, top_as_buffer)
}
