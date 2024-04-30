mod base;
mod utils;

extern crate nalgebra_glm as glm;

use ash::vk::{self, CommandBufferUsageFlags};
use base::{AppBase, GlobalUniforms};
use bytemuck::bytes_of;
use glm::{infinite_perspective_rh_zo, inverse};
use utils::{HEIGHT, WIDTH};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard,
    platform::run_on_demand::EventLoopExtRunOnDemand,
    window::Window,
};

fn update_camera(base: &mut AppBase, command_buffer: vk::CommandBuffer) {
    let elapsed = base.start.elapsed().as_secs_f32();
    base.camera.translation.y = elapsed.sin();
    base.camera.translation.x = elapsed.cos();

    let global_yaw = base.camera.global_rotation.x;
    let global_pitch = base.camera.global_rotation.y;
    let global_roll = base.camera.global_rotation.z;

    let local_yaw = base.camera.local_rotation.x;
    let local_pitch = base.camera.local_rotation.y;
    let local_roll = base.camera.local_rotation.z;

    let local_view = glm::Mat4::from_euler_angles(local_roll, local_pitch, local_yaw);
    let view_matrix = local_view
        * glm::Mat4::from_euler_angles(global_roll, global_pitch, global_yaw)
            .prepend_translation(&-base.camera.translation);

    let proj_matrix = infinite_perspective_rh_zo(base.aspect_ratio(), glm::pi::<f32>() / 2.5, 0.1);

    let view_proj = view_matrix * proj_matrix;
    let view_inverse = inverse(&view_matrix);
    let proj_inverse = inverse(&proj_matrix);

    let uniform_buffer_data = GlobalUniforms {
        view_proj,
        view_inverse,
        proj_inverse,
    };

    let uniforms_buffer = base.uniforms_buffer.as_mut().unwrap();

    unsafe {
        let buffer_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .buffer(uniforms_buffer.buffer)
            .size(uniforms_buffer.size);

        base.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::DEVICE_GROUP,
            &[],
            &[buffer_barrier],
            &[],
        );
    }

    unsafe {
        base.device.cmd_update_buffer(
            command_buffer,
            uniforms_buffer.buffer,
            0,
            bytes_of(&uniform_buffer_data),
        )
    }

    unsafe {
        let buffer_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .buffer(uniforms_buffer.buffer)
            .size(uniforms_buffer.size);

        base.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::DEVICE_GROUP,
            &[],
            &[buffer_barrier],
            &[],
        );
    }
}

fn toggle_capture_mouse(base: &mut AppBase) {
    if base.focused {
        base.focused = false;
        base.window
            .set_cursor_grab(winit::window::CursorGrabMode::None)
            .unwrap();
        base.window.set_cursor_visible(true);
    } else {
        base.focused = true;
        base.window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .unwrap();
        base.window.set_cursor_visible(false);
    }
}

#[derive(Default)]
struct App<'a> {
    window: Option<Window>,
    base: Option<AppBase<'a>>,
    rt_command_buffer: Option<vk::CommandBuffer>,
    frame_start: Option<std::time::Instant>,
}

impl App<'static> {
    fn main_loop(&mut self) {
        let base = self.base.as_mut().unwrap();
        let rt_command_buffer = self.rt_command_buffer.unwrap();
        unsafe {
            base.device
                .wait_for_fences(&[base.swapchain_acquire_fence], true, std::u64::MAX)
                .expect("Wait for fence failed.");

            base.device
                .reset_fences(&[base.swapchain_acquire_fence])
                .expect("Reset fences failed.");
        }

        if base.resized {
            base.recreate_swapchain().unwrap();
            base.resized = false;
        }

        base.current_frames_counter += 1;

        self.frame_start = Some(std::time::Instant::now());

        base.update_delta_time();

        if self
            .frame_start
            .unwrap()
            .duration_since(base.last_second)
            .as_secs()
            > 0
        {
            // println!("{}fps", base.current_frames_counter);
            base.reset_fps_counter();
        }

        let (present_index, _) = unsafe {
            base.swapchain_loader.acquire_next_image(
                base.swapchain.unwrap(),
                std::u64::MAX,
                base.present_complete_semaphore,
                base.swapchain_acquire_fence,
            )
        }
        .unwrap();

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
                    rt_command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Reset command buffer failed.");

            let rt_command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            base.device
                .begin_command_buffer(rt_command_buffer, &rt_command_buffer_begin_info)
                .expect("Begin commandbuffer");

            update_camera(base, rt_command_buffer);

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

            let command_buffers = vec![rt_command_buffer];
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
            .wait_semaphores(&wait_semaphors)
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

        base.window.request_redraw();
    }
}

impl ApplicationHandler for App<'static> {
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.base.is_none() {
            let mut app = AppBase::new(&event_loop, WIDTH, HEIGHT);
            app.init();

            let rt_command_buffer = {
                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .command_pool(app.pool)
                    .level(vk::CommandBufferLevel::PRIMARY);

                unsafe {
                    app.device
                        .allocate_command_buffers(&command_buffer_allocate_info)
                }
                .expect("Failed to allocate Command Buffers!")[0]
            };

            self.rt_command_buffer = Some(rt_command_buffer);

            unsafe {
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                app.device
                    .begin_command_buffer(rt_command_buffer, &begin_info)
                    .unwrap();
            }

            unsafe {
                app.device.end_command_buffer(rt_command_buffer).unwrap();
            }

            self.base = Some(app);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: keyboard::Key::Named(keyboard::NamedKey::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(_) => self.base.as_mut().unwrap().resized = true,
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => toggle_capture_mouse(self.base.as_mut().unwrap()),
            // Event::DeviceEvent {
            //     event: DeviceEvent::MouseMotion { delta },
            //     ..
            // } => base.update_look_position(delta),
            WindowEvent::RedrawRequested => self.main_loop(),
            _ => (),
        };
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.base.as_mut().unwrap().update_look_position(delta)
            }
            _ => (),
        };
    }
}

fn main() {
    let mut event_loop = EventLoop::new().unwrap();

    let mut app = App::default();

    event_loop.run_app_on_demand(&mut app).unwrap();
}
