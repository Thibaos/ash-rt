mod acceleration_structure;
mod base;
mod utils;

use acceleration_structure::create_bottom_as;
use acceleration_structure::create_geometry;
use acceleration_structure::create_instances;
use acceleration_structure::create_top_as;
use ash::prelude::VkResult;
use ash::util::Align;
use ash::vk;
use ash::vk::PipelineStageFlags;
use base::ExampleBase;
use std::ptr;
use utils::create_rt_descriptor_sets;
use utils::create_rt_sbt;
use utils::init_rt;
use utils::HEIGHT;
use utils::WIDTH;

unsafe fn create_shader_module(device: &ash::Device, code: &[u8]) -> VkResult<vk::ShaderModule> {
    let shader_module_create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ShaderModuleCreateFlags::empty(),
        code_size: code.len(),
        p_code: code.as_ptr() as *const u32,
    };

    device.create_shader_module(&shader_module_create_info, None)
}

fn aligned_size(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

fn get_memory_type_index(
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

unsafe fn get_buffer_device_address(device: &ash::Device, buffer: vk::Buffer) -> u64 {
    let buffer_device_address_info = vk::BufferDeviceAddressInfo::builder()
        .buffer(buffer)
        .build();

    device.get_buffer_device_address(&buffer_device_address_info)
}

#[derive(Clone)]
pub struct BufferResource {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl BufferResource {
    fn new(
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
        device: &ash::Device,
        device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();

            let buffer = device.create_buffer(&buffer_info, None).unwrap();

            let memory_req = device.get_buffer_memory_requirements(buffer);

            let memory_index = get_memory_type_index(
                device_memory_properties,
                memory_req.memory_type_bits,
                memory_properties,
            );

            let mut memory_allocate_flags_info = vk::MemoryAllocateFlagsInfo::builder()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS)
                .build();

            let mut allocate_info_builder = vk::MemoryAllocateInfo::builder();

            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                allocate_info_builder =
                    allocate_info_builder.push_next(&mut memory_allocate_flags_info);
            }

            let allocate_info = allocate_info_builder
                .allocation_size(memory_req.size)
                .memory_type_index(memory_index)
                .build();

            let memory = device.allocate_memory(&allocate_info, None).unwrap();

            device.bind_buffer_memory(buffer, memory, 0).unwrap();

            BufferResource {
                buffer,
                memory,
                size,
            }
        }
    }

    fn store<T: Copy>(&mut self, data: &[T], device: &ash::Device) {
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

    unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None);
    }
}

fn main() {
    let base = ExampleBase::new(WIDTH, HEIGHT);

    let acceleration_structure =
        ash::extensions::khr::AccelerationStructure::new(&base.instance, &base.device);

    let graphics_queue = unsafe { base.device.get_device_queue(base.queue_family_index, 0) };

    let device_memory_properties = unsafe {
        base.instance
            .get_physical_device_memory_properties(base.physical_device)
    };

    let rt_image = {
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(base.surface_format.format)
            .extent(
                vk::Extent3D::builder()
                    .width(WIDTH)
                    .height(HEIGHT)
                    .depth(1)
                    .build(),
            )
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .build();

        unsafe { base.device.create_image(&image_create_info, None) }.unwrap()
    };

    let rt_image_memory = {
        let mem_reqs = unsafe { base.device.get_image_memory_requirements(rt_image) };
        let mem_alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_reqs.size)
            .memory_type_index(get_memory_type_index(
                device_memory_properties,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ));

        unsafe { base.device.allocate_memory(&mem_alloc_info, None) }.unwrap()
    };

    unsafe { base.device.bind_image_memory(rt_image, rt_image_memory, 0) }.unwrap();

    let rt_image_view = {
        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(base.surface_format.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(rt_image)
            .build();

        unsafe { base.device.create_image_view(&image_view_create_info, None) }.unwrap()
    };

    // rt image to general layout
    {
        let command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(base.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .build();

            let command_buffers =
                unsafe { base.device.allocate_command_buffers(&allocate_info) }.unwrap();
            command_buffers[0]
        };

        unsafe {
            base.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build(),
            )
        }
        .unwrap();

        let image_barrier = vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(rt_image)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .build();

        unsafe {
            base.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier],
            );

            base.device.end_command_buffer(command_buffer).unwrap();
        }

        let command_buffers = [command_buffer];

        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .build()];

        unsafe {
            base.device
                .queue_submit(graphics_queue, &submit_infos, vk::Fence::null())
                .expect("Failed to execute queue submit.");

            base.device.queue_wait_idle(graphics_queue).unwrap();
            base.device
                .free_command_buffers(base.pool, &[command_buffer]);
        }
    }

    let (geometry, aabb_buffer) = create_geometry(&base, device_memory_properties);

    let (bottom_as, bottom_as_buffer) = create_bottom_as(
        &acceleration_structure,
        geometry,
        &base.device,
        device_memory_properties,
        base.pool,
        graphics_queue,
    );

    let (instance_count, instance_buffer) = create_instances(
        &acceleration_structure,
        bottom_as,
        &base.device,
        device_memory_properties,
    );

    let (top_as, top_as_buffer) = create_top_as(
        &acceleration_structure,
        instance_count,
        &instance_buffer,
        &base.device,
        device_memory_properties,
        base.pool,
        graphics_queue,
    );

    let (rt_pipeline, rt_pipeline_properties) = init_rt(&base);

    let (
        descriptor_pool,
        descriptor_set,
        descriptor_set_layout,
        pipeline,
        pipeline_layout,
        shader_group_count,
    ) = create_rt_descriptor_sets(&base, &rt_pipeline, rt_image_view, top_as);

    let (
        shader_binding_table_buffer,
        sbt_raygen_region,
        sbt_miss_region,
        sbt_hit_region,
        sbt_call_region,
    ) = create_rt_sbt(
        &base,
        &rt_pipeline,
        &rt_pipeline_properties,
        device_memory_properties,
        pipeline,
        shader_group_count,
    );

    let renderpass_attachments = [
        vk::AttachmentDescription {
            format: base.surface_format.format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        },
        vk::AttachmentDescription {
            format: vk::Format::D16_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        },
    ];
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_attachment_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..Default::default()
    }];

    let subpass = vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

    let renderpass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&renderpass_attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&dependencies);

    let renderpass = unsafe {
        base.device
            .create_render_pass(&renderpass_create_info, None)
    }
    .unwrap();

    let framebuffers: Vec<vk::Framebuffer> = base
        .present_image_views
        .iter()
        .map(|&present_image_view| {
            let framebuffer_attachments = [present_image_view, base.depth_image_view];
            let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass)
                .attachments(&framebuffer_attachments)
                .width(base.surface_resolution.width)
                .height(base.surface_resolution.height)
                .layers(1);

            unsafe {
                base.device
                    .create_framebuffer(&frame_buffer_create_info, None)
            }
            .unwrap()
        })
        .collect();

    let rt_command_buffer = {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(base.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .build();

        unsafe {
            base.device
                .allocate_command_buffers(&command_buffer_allocate_info)
        }
        .expect("Failed to allocate Command Buffers!")[0]
    };

    base.render_loop(|| {
        let (present_index, _) = unsafe {
            base.swapchain_loader.acquire_next_image(
                base.swapchain,
                std::u64::MAX,
                base.present_complete_semaphore,
                vk::Fence::null(),
            )
        }
        .unwrap();

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(renderpass)
            .framebuffer(framebuffers[present_index as usize])
            .render_area(base.surface_resolution.into())
            .clear_values(&clear_values);

        let current_swapchain_image = base.present_images[present_index as usize];

        unsafe {
            base.device
                .wait_for_fences(&[base.draw_commands_reuse_fence], true, std::u64::MAX)
                .expect("Wait for fence failed.");

            base.device
                .reset_fences(&[base.draw_commands_reuse_fence])
                .expect("Reset fences failed.");

            base.device
                .reset_command_buffer(
                    base.draw_command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Reset command buffer failed.");

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            base.device
                .begin_command_buffer(base.draw_command_buffer, &command_buffer_begin_info)
                .expect("Begin commandbuffer");

            {
                base.device.cmd_begin_render_pass(
                    base.draw_command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                base.device.cmd_end_render_pass(base.draw_command_buffer);
            }

            base.device
                .end_command_buffer(base.draw_command_buffer)
                .expect("End commandbuffer");

            base.device
                .reset_command_buffer(
                    rt_command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Reset command buffer failed.");

            let rt_command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            base.device
                .begin_command_buffer(rt_command_buffer, &rt_command_buffer_begin_info)
                .expect("Begin commandbuffer");

            // full rt pass
            {
                base.device.cmd_bind_pipeline(
                    rt_command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    pipeline,
                );
                base.device.cmd_bind_descriptor_sets(
                    rt_command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    pipeline_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
                rt_pipeline.cmd_trace_rays(
                    rt_command_buffer,
                    &sbt_raygen_region,
                    &sbt_miss_region,
                    &sbt_hit_region,
                    &sbt_call_region,
                    WIDTH,
                    HEIGHT,
                    1,
                );
            }

            // current swapchain to dst layout
            {
                let image_barrier = vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(
                        vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(current_swapchain_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .build();

                base.device.cmd_pipeline_barrier(
                    rt_command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier],
                );
            }

            // rt image to src layout
            {
                let image_barrier = vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ,
                    )
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(rt_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .build();

                base.device.cmd_pipeline_barrier(
                    rt_command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier],
                );
            }

            // copy rt image to swapchain
            {
                let copy_region = vk::ImageCopy::builder()
                    .src_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .build(),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .build(),
                    )
                    .extent(
                        vk::Extent3D::builder()
                            .width(WIDTH)
                            .height(HEIGHT)
                            .depth(1)
                            .build(),
                    )
                    .build();

                base.device.cmd_copy_image(
                    rt_command_buffer,
                    rt_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    current_swapchain_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_region],
                );
            }

            // current swapchain to present src layout
            {
                let image_barrier = vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(current_swapchain_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .build();

                base.device.cmd_pipeline_barrier(
                    rt_command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier],
                );
            }

            // rt image back to general layout
            {
                let image_barrier = vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(rt_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .build();

                base.device.cmd_pipeline_barrier(
                    rt_command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier],
                );
            }

            base.device
                .end_command_buffer(rt_command_buffer)
                .expect("End commandbuffer");

            let command_buffers = vec![base.draw_command_buffer, rt_command_buffer];
            let wait_semaphores = &[base.present_complete_semaphore];
            let signal_semaphores = &[base.rendering_complete_semaphore];

            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(&[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&command_buffers)
                .signal_semaphores(signal_semaphores);

            base.device
                .queue_submit(
                    base.present_queue,
                    &[submit_info.build()],
                    base.draw_commands_reuse_fence,
                )
                .expect("queue submit failed.");
        }

        let wait_semaphors = [base.rendering_complete_semaphore];
        let swapchains = [base.swapchain];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            base.swapchain_loader
                .queue_present(base.present_queue, &present_info)
        }
        .unwrap();
    });

    unsafe {
        base.device.device_wait_idle().unwrap();
    }

    unsafe {
        base.device.destroy_descriptor_pool(descriptor_pool, None);
        shader_binding_table_buffer.destroy(&base.device);
        base.device.destroy_pipeline(pipeline, None);
        base.device
            .destroy_descriptor_set_layout(descriptor_set_layout, None);
        base.device.destroy_pipeline_layout(pipeline_layout, None);
    }

    unsafe {
        acceleration_structure.destroy_acceleration_structure(bottom_as, None);
        bottom_as_buffer.destroy(&base.device);

        acceleration_structure.destroy_acceleration_structure(top_as, None);
        top_as_buffer.destroy(&base.device);

        base.device.destroy_image_view(rt_image_view, None);
        base.device.destroy_image(rt_image, None);
        base.device.free_memory(rt_image_memory, None);
    }

    unsafe {
        instance_buffer.destroy(&base.device);
        aabb_buffer.destroy(&base.device);
    }

    unsafe {
        for framebuffer in framebuffers {
            base.device.destroy_framebuffer(framebuffer, None);
        }
        base.device.destroy_render_pass(renderpass, None);
    }
}
