mod base;
mod utils;

extern crate nalgebra_glm as glm;

use ash::vk;
use base::{AppBase, GlobalUniforms};
use bytemuck::bytes_of;
use glm::{infinite_perspective_rh_zo, inverse, look_at_rh, Vec3};
use utils::{HEIGHT, WIDTH};
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard,
    platform::run_on_demand::EventLoopExtRunOnDemand,
};

fn main() {
    let mut event_loop = EventLoop::new().unwrap();

    let mut base = AppBase::new(&event_loop, WIDTH, HEIGHT);
    base.init();

    let rt_command_buffer = {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(base.pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            base.device
                .allocate_command_buffers(&command_buffer_allocate_info)
        }
        .expect("Failed to allocate Command Buffers!")[0]
    };

    let eye = Vec3::new(0.0, 0.0, -2.0);
    let target = Vec3::new(0.0, 0.0, 1.0);

    let view_matrix = look_at_rh(&eye, &target, &Vec3::y());

    let proj_matrix = infinite_perspective_rh_zo(WIDTH as f32 / HEIGHT as f32, 3.14 / 2.0, 0.1);

    let view_proj = view_matrix * proj_matrix;
    let view_inverse = inverse(&view_matrix);
    let proj_inverse = inverse(&proj_matrix);

    let uniform_buffer_data = GlobalUniforms {
        origin: eye.to_homogeneous().into(),
        direction: target.to_homogeneous().into(),
        view_proj,
        view_inverse,
        proj_inverse,
    };

    base.uniforms_buffer
        .as_mut()
        .unwrap()
        .store(bytes_of(&uniform_buffer_data), &base.device);

    let main_loop = |base: &mut AppBase<'_>| {
        if base.resized {
            base.recreate_swapchain().unwrap();
            base.resized = false;
        }

        let (present_index, _) = unsafe {
            base.swapchain_loader.acquire_next_image(
                base.swapchain.unwrap(),
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

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(base.render_pass.unwrap())
            .framebuffer(base.framebuffers.as_ref().unwrap()[present_index as usize])
            .render_area(base.surface_resolution.into())
            .clear_values(&clear_values);

        let current_swapchain_image = base.present_images.as_ref().unwrap()[present_index as usize];

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

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
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

            let rt_command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            base.device
                .begin_command_buffer(rt_command_buffer, &rt_command_buffer_begin_info)
                .expect("Begin commandbuffer");

            // full rt pass
            {
                base.device.cmd_bind_pipeline(
                    rt_command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    base.pipeline.unwrap(),
                );
                base.device.cmd_bind_descriptor_sets(
                    rt_command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    base.pipeline_layout.unwrap(),
                    0,
                    &[
                        base.rt_descriptor_set.unwrap(),
                        base.uniforms_descriptor_set.unwrap(),
                    ],
                    &[],
                );
                base.ray_tracing_pipeline_loader.cmd_trace_rays(
                    rt_command_buffer,
                    &base.sbt_raygen_region.unwrap(),
                    &base.sbt_miss_region.unwrap(),
                    &base.sbt_hit_region.unwrap(),
                    &base.sbt_call_region.unwrap(),
                    base.surface_resolution.width,
                    base.surface_resolution.height,
                    1,
                );
            }

            // current swapchain to dst layout
            {
                let image_barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(
                        vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(current_swapchain_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

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
                let image_barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ,
                    )
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(base.rt_image.unwrap())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

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
                let copy_region = vk::ImageCopy::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .extent(
                        vk::Extent3D::default()
                            .width(base.surface_resolution.width)
                            .height(base.surface_resolution.height)
                            .depth(1),
                    );

                base.device.cmd_copy_image(
                    rt_command_buffer,
                    base.rt_image.unwrap(),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    current_swapchain_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_region],
                );
            }

            // current swapchain to present src layout
            {
                let image_barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(current_swapchain_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

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
                let image_barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(base.rt_image.unwrap())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

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

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&command_buffers)
                .signal_semaphores(signal_semaphores);

            base.device
                .queue_submit(
                    base.present_queue,
                    &[submit_info],
                    base.draw_commands_reuse_fence,
                )
                .expect("queue submit failed.");
        }

        let wait_semaphors = [base.rendering_complete_semaphore];
        let swapchains = [base.swapchain.unwrap()];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            if let Err(code) = base
                .swapchain_loader
                .queue_present(base.present_queue, &present_info)
            {
                match code {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => base.recreate_swapchain().unwrap(),
                    _ => (),
                }
            }
        }
    };

    event_loop
        .run_on_demand(|event, window_target| match event {
            Event::WindowEvent {
                event:
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                logical_key: keyboard::Key::Named(keyboard::NamedKey::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => window_target.exit(),
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => base.resized = true,
            Event::AboutToWait => main_loop(&mut base),
            _ => (),
        })
        .unwrap();
}
