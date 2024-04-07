use ash::{
    extensions::khr::RayTracingPipeline,
    vk::{
        self, AccelerationStructureKHR, DescriptorPool, DescriptorSet, DescriptorSetLayout,
        ImageView, PhysicalDeviceMemoryProperties, PhysicalDeviceRayTracingPipelinePropertiesKHR,
        Pipeline, PipelineLayout, StridedDeviceAddressRegionKHR,
    },
};

use crate::{
    aligned_size, base::ExampleBase, create_shader_module, get_buffer_device_address,
    BufferResource,
};

pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;

pub fn init_rt(
    base: &ExampleBase,
) -> (
    RayTracingPipeline,
    PhysicalDeviceRayTracingPipelinePropertiesKHR,
) {
    let rt_pipeline = ash::extensions::khr::RayTracingPipeline::new(&base.instance, &base.device);

    let mut rt_pipeline_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();

    {
        let mut physical_device_properties2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut rt_pipeline_properties)
            .build();

        unsafe {
            base.instance.get_physical_device_properties2(
                base.physical_device,
                &mut physical_device_properties2,
            );
        }
    }

    (rt_pipeline, rt_pipeline_properties)
}

pub fn create_rt_descriptor_sets(
    base: &ExampleBase,
    rt_pipeline: &RayTracingPipeline,
    image_view: ImageView,
    top_as: AccelerationStructureKHR,
) -> (
    DescriptorPool,
    DescriptorSet,
    DescriptorSetLayout,
    Pipeline,
    PipelineLayout,
    usize,
) {
    let mut count_allocate_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
        .descriptor_counts(&[1])
        .build();

    let descriptor_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
        },
    ];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
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
    ];

    let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
        .binding_flags(&binding_flags_inner)
        .build();

    let descriptor_set_layout = unsafe {
        base.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&[
                    vk::DescriptorSetLayoutBinding::builder()
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                        .binding(0)
                        .build(),
                    vk::DescriptorSetLayoutBinding::builder()
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                        .binding(1)
                        .build(),
                ])
                .push_next(&mut binding_flags)
                .build(),
            None,
        )
    }
    .unwrap();

    const SHADER: &[u8] = include_bytes!(env!("shaders.spv"));

    let shader_module = unsafe { create_shader_module(&base.device, SHADER).unwrap() };

    let layouts = vec![descriptor_set_layout];
    let layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);

    let pipeline_layout = unsafe {
        base.device
            .create_pipeline_layout(&layout_create_info, None)
    }
    .unwrap();

    let shader_groups = vec![
        // group0 = [ raygen ]
        vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .build(),
        // group1 = [ chit ]
        vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(1)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(2)
            .build(),
        // group3 = [ miss ]
        vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(3)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .build(),
    ];

    let shader_stages = vec![
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main_ray_generation\0").unwrap())
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main_closest_hit\0").unwrap())
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main_intersection\0").unwrap())
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main_miss\0").unwrap())
            .build(),
    ];

    let pipeline = unsafe {
        rt_pipeline.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &[vk::RayTracingPipelineCreateInfoKHR::builder()
                .stages(&shader_stages)
                .groups(&shader_groups)
                .max_pipeline_ray_recursion_depth(1)
                .layout(pipeline_layout)
                .build()],
            None,
        )
    }
    .unwrap()[0];

    unsafe {
        base.device.destroy_shader_module(shader_module, None);
    }

    let descriptor_set = unsafe {
        base.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_set_layout])
                .push_next(&mut count_allocate_info)
                .build(),
        )
    }
    .unwrap()[0];

    let accel_structs = [top_as];

    let mut accel_info = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
        .acceleration_structures(&accel_structs)
        .build();

    let mut accel_write = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .push_next(&mut accel_info)
        .build();

    accel_write.descriptor_count = 1;

    let image_info = [vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(image_view)
        .build()];

    let image_write = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(1)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(&image_info)
        .build();

    unsafe {
        base.device
            .update_descriptor_sets(&[accel_write, image_write], &[]);
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
    rt_pipeline: &RayTracingPipeline,
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
            rt_pipeline.get_ray_tracing_shader_group_handles(
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

    let sbt_raygen_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(sbt_address)
        .size(handle_size_aligned)
        .stride(handle_size_aligned)
        .build();

    let sbt_miss_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(sbt_address + 2 * handle_size_aligned)
        .size(handle_size_aligned)
        .stride(handle_size_aligned)
        .build();

    let sbt_hit_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(sbt_address + handle_size_aligned)
        .size(handle_size_aligned)
        .stride(handle_size_aligned)
        .build();

    let sbt_call_region = vk::StridedDeviceAddressRegionKHR::default();

    (
        shader_binding_table_buffer,
        sbt_raygen_region,
        sbt_miss_region,
        sbt_hit_region,
        sbt_call_region,
    )
}
