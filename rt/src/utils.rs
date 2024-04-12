use ash::{
    util::Align,
    vk::{
        self, AccelerationStructureKHR, DescriptorPool, DescriptorSet, DescriptorSetLayout,
        ImageView, PhysicalDeviceMemoryProperties, PhysicalDeviceRayTracingPipelinePropertiesKHR,
        Pipeline, PipelineLayout, StridedDeviceAddressRegionKHR,
    },
};

use crate::{aligned_size, base::ExampleBase, create_shader_module, get_buffer_device_address};

pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;

#[derive(Clone, Debug, Copy)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

pub fn get_memory_type_index(
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    mut type_bits: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    for i in 0..device_memory_properties.memory_type_count {
        if (type_bits & 1) == 1
            && (device_memory_properties.memory_types[i as usize].property_flags & properties)
                == properties
        {
            return i;
        }
        type_bits >>= 1;
    }
    0
}

#[derive(Clone)]
pub struct BufferResource {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl BufferResource {
    pub fn new(
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
        device: &ash::Device,
        device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = device.create_buffer(&buffer_info, None).unwrap();

            let memory_req = device.get_buffer_memory_requirements(buffer);

            let memory_index = get_memory_type_index(
                device_memory_properties,
                memory_req.memory_type_bits,
                memory_properties,
            );

            let mut memory_allocate_flags_info = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

            let mut allocate_info_builder = vk::MemoryAllocateInfo::default();

            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                allocate_info_builder =
                    allocate_info_builder.push_next(&mut memory_allocate_flags_info);
            }

            let allocate_info = allocate_info_builder
                .allocation_size(memory_req.size)
                .memory_type_index(memory_index);

            let memory = device.allocate_memory(&allocate_info, None).unwrap();

            device.bind_buffer_memory(buffer, memory, 0).unwrap();

            BufferResource {
                buffer,
                memory,
                size,
            }
        }
    }

    pub fn store<T: Copy>(&mut self, data: &[T], device: &ash::Device) {
        unsafe {
            let size = std::mem::size_of_val(data) as u64;
            assert!(
                self.size >= size,
                "Data size is larger than buffer size ({}, {}).",
                self.size,
                size
            );
            let mapped_ptr = self.map(size, device);
            let mut mapped_slice = Align::new(mapped_ptr, std::mem::align_of::<T>() as u64, size);
            mapped_slice.copy_from_slice(data);
            self.unmap(device);
        }
    }

    fn map(&mut self, size: vk::DeviceSize, device: &ash::Device) -> *mut std::ffi::c_void {
        unsafe {
            let data: *mut std::ffi::c_void = device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            data
        }
    }

    fn unmap(&mut self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.memory);
        }
    }

    pub unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None);
    }
}

pub fn create_color_buffers(
    base: &ExampleBase,
    device_memory_properties: PhysicalDeviceMemoryProperties,
) -> BufferResource {
    let colors = [
        Vector3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
            _pad: 0.0,
        },
        Vector3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
            _pad: 0.0,
        },
        Vector3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
            _pad: 0.0,
        },
    ];

    let mut colors_buffer = BufferResource::new(
        std::mem::size_of_val(&colors) as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        &base.device,
        device_memory_properties,
    );
    colors_buffer.store(&colors, &base.device);

    colors_buffer
}

pub fn create_rt_descriptor_sets(
    base: &ExampleBase,
    image_view: ImageView,
    top_as: AccelerationStructureKHR,
    colors_buffers: &BufferResource,
) -> (
    DescriptorPool,
    DescriptorSet,
    DescriptorSetLayout,
    Pipeline,
    PipelineLayout,
    usize,
) {
    let mut count_allocate_info =
        vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(&[1]);

    let descriptor_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        },
    ];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&descriptor_sizes)
        .max_sets(1);

    let descriptor_pool = unsafe {
        base.device
            .create_descriptor_pool(&descriptor_pool_info, None)
    }
    .unwrap();

    let binding_flags_inner = [
        vk::DescriptorBindingFlagsEXT::empty(),
        vk::DescriptorBindingFlagsEXT::empty(),
        vk::DescriptorBindingFlags::empty(),
    ];

    let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
        .binding_flags(&binding_flags_inner);

    let descriptor_set_layout = unsafe {
        base.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&[
                    vk::DescriptorSetLayoutBinding::default()
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                        .binding(0),
                    vk::DescriptorSetLayoutBinding::default()
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                        .binding(1),
                    vk::DescriptorSetLayoutBinding::default()
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                        .binding(2),
                ])
                .push_next(&mut binding_flags),
            None,
        )
    }
    .unwrap();

    const SHADER: &[u8] = include_bytes!(env!("shaders.spv"));

    let shader_module = unsafe { create_shader_module(&base.device, SHADER).unwrap() };

    let layouts = vec![descriptor_set_layout];
    let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

    let pipeline_layout = unsafe {
        base.device
            .create_pipeline_layout(&layout_create_info, None)
    }
    .unwrap();

    let shader_groups = vec![
        // group0 = [ raygen ]
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        // group1 = [ chit ]
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(1)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(2),
        // group3 = [ miss ]
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(3)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
    ];

    let shader_stages = vec![
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main_ray_generation\0").unwrap()),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main_closest_hit\0").unwrap()),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main_intersection\0").unwrap()),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main_miss\0").unwrap()),
    ];

    let pipeline = unsafe {
        base.ray_tracing_pipeline_loader
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[vk::RayTracingPipelineCreateInfoKHR::default()
                    .stages(&shader_stages)
                    .groups(&shader_groups)
                    .max_pipeline_ray_recursion_depth(1)
                    .layout(pipeline_layout)],
                None,
            )
    }
    .unwrap()[0];

    unsafe {
        base.device.destroy_shader_module(shader_module, None);
    }

    let descriptor_set = unsafe {
        base.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_set_layout])
                .push_next(&mut count_allocate_info),
        )
    }
    .unwrap()[0];

    let accel_structs = [top_as];

    let mut accel_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
        .acceleration_structures(&accel_structs);

    let accel_write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .descriptor_count(1)
        .push_next(&mut accel_info);

    let image_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(image_view)];

    let image_write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(1)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(&image_info);

    let colors_buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(colors_buffers.buffer)
        .range(vk::WHOLE_SIZE)];

    let colors_buffer_write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(2)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&colors_buffer_info);

    unsafe {
        base.device
            .update_descriptor_sets(&[accel_write, image_write, colors_buffer_write], &[]);
    }

    (
        descriptor_pool,
        descriptor_set,
        descriptor_set_layout,
        pipeline,
        pipeline_layout,
        shader_groups.len(),
    )
}

pub fn create_rt_sbt(
    base: &ExampleBase,
    rt_pipeline_properties: &PhysicalDeviceRayTracingPipelinePropertiesKHR,
    device_memory_properties: PhysicalDeviceMemoryProperties,
    pipeline: Pipeline,
    shader_group_count: usize,
) -> (
    BufferResource,
    StridedDeviceAddressRegionKHR,
    StridedDeviceAddressRegionKHR,
    StridedDeviceAddressRegionKHR,
    StridedDeviceAddressRegionKHR,
) {
    let handle_size_aligned = aligned_size(
        rt_pipeline_properties.shader_group_handle_size,
        rt_pipeline_properties.shader_group_base_alignment,
    ) as u64;

    let shader_binding_table_buffer = {
        let incoming_table_data = unsafe {
            base.ray_tracing_pipeline_loader
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    shader_group_count as u32,
                    shader_group_count * rt_pipeline_properties.shader_group_handle_size as usize,
                )
        }
        .unwrap();

        let table_size = shader_group_count * handle_size_aligned as usize;
        let mut table_data = vec![0u8; table_size];

        for i in 0..shader_group_count {
            table_data[i * handle_size_aligned as usize
                ..i * handle_size_aligned as usize
                    + rt_pipeline_properties.shader_group_handle_size as usize]
                .copy_from_slice(
                    &incoming_table_data[i * rt_pipeline_properties.shader_group_handle_size
                        as usize
                        ..i * rt_pipeline_properties.shader_group_handle_size as usize
                            + rt_pipeline_properties.shader_group_handle_size as usize],
                );
        }

        let mut shader_binding_table_buffer = BufferResource::new(
            table_size as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            &base.device,
            device_memory_properties,
        );

        shader_binding_table_buffer.store(&table_data, &base.device);

        shader_binding_table_buffer
    };

    // |[ raygen shader ]|[ hit shader  ]|[ miss shader ]|
    // |                 |               |               |
    // | 0               | 1             | 2             | 3

    let sbt_address =
        unsafe { get_buffer_device_address(&base.device, shader_binding_table_buffer.buffer) };

    let sbt_raygen_region = vk::StridedDeviceAddressRegionKHR::default()
        .device_address(sbt_address)
        .size(handle_size_aligned)
        .stride(handle_size_aligned);

    let sbt_miss_region = vk::StridedDeviceAddressRegionKHR::default()
        .device_address(sbt_address + 2 * handle_size_aligned)
        .size(handle_size_aligned)
        .stride(handle_size_aligned);

    let sbt_hit_region = vk::StridedDeviceAddressRegionKHR::default()
        .device_address(sbt_address + handle_size_aligned)
        .size(handle_size_aligned)
        .stride(handle_size_aligned);

    let sbt_call_region = vk::StridedDeviceAddressRegionKHR::default();

    (
        shader_binding_table_buffer,
        sbt_raygen_region,
        sbt_miss_region,
        sbt_hit_region,
        sbt_call_region,
    )
}
