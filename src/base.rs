use ash::vk;
use bevy_transform::components::Transform;
use bytemuck::bytes_of;
use std::default::Default;

use winit::event_loop::ActiveEventLoop;

use crate::{
    player_controller::PlayerController,
    uniform_types::{CameraTransform, GlobalUniforms},
    utils::WIDTH,
    vk_controller::VkController,
};

pub struct AppBase<'a> {
    pub vk_controller: VkController<'a>,

    pub resized: bool,
    pub focused: bool,

    pub frame_start: std::time::Instant,
    pub current_frames_counter: u64,
    pub last_second: std::time::Instant,
    pub last_frame_update: std::time::Instant,
    pub delta_time: std::time::Duration,

    pub player_controller: PlayerController,
    pub camera: CameraTransform,
    pub sensitivity: f64,
}

impl AppBase<'_> {
    pub fn aspect_ratio(&self) -> f32 {
        self.vk_controller.surface_resolution.width as f32
            / self.vk_controller.surface_resolution.height as f32
    }

    pub fn reset_fps_counter(&mut self) {
        self.last_second = std::time::Instant::now();
        self.current_frames_counter = 0;
    }

    pub fn update_delta_time(&mut self) {
        let now = std::time::Instant::now();
        let delta = now.duration_since(self.last_frame_update);
        self.last_frame_update = now;
        self.delta_time = delta;
    }

    pub fn update_look_position(&mut self, delta: (f64, f64)) {
        if self.focused {
            let (mut yaw, mut pitch, _) = self
                .camera
                .transform
                .rotation
                .to_euler(bevy_math::EulerRot::YXZ);

            let scale = self.vk_controller.window.scale_factor();

            let surface_size =
                self.vk_controller
                    .surface_resolution
                    .height
                    .min(self.vk_controller.surface_resolution.width) as f64
                    * scale
                    / WIDTH as f64;

            yaw -= (delta.0 * self.sensitivity / surface_size) as f32;
            pitch -= (delta.1 * self.sensitivity / surface_size) as f32;

            pitch = pitch.clamp(-glm::half_pi::<f32>(), glm::half_pi());

            self.camera.transform.rotation =
                bevy_math::Quat::from_axis_angle(bevy_math::Vec3::Y, yaw)
                    * bevy_math::Quat::from_axis_angle(bevy_math::Vec3::X, pitch);
        }
    }

    pub fn toggle_capture_mouse(&mut self) {
        if self.focused {
            self.focused = false;
            self.vk_controller
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
            self.vk_controller.window.set_cursor_visible(true);
        } else {
            self.focused = true;
            self.vk_controller
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .unwrap();
            self.vk_controller.window.set_cursor_visible(false);
        }
    }

    pub fn update_camera(&mut self) {
        let local_z = self.camera.transform.local_z();
        let forward = -bevy_math::Vec3::new(local_z.x, 0.0, local_z.z);
        let right = bevy_math::Vec3::new(local_z.z, 0.0, -local_z.x);

        let mut velocity = bevy_math::Vec3::ZERO;

        if self.player_controller.forward {
            velocity += forward;
        } else if self.player_controller.backward {
            velocity -= forward;
        }
        if self.player_controller.left {
            velocity -= right;
        } else if self.player_controller.right {
            velocity += right;
        }
        if self.player_controller.up {
            velocity += bevy_math::Vec3::Y;
        } else if self.player_controller.down {
            velocity -= bevy_math::Vec3::Y;
        }

        velocity = velocity.normalize_or_zero();

        self.camera.transform.translation +=
            velocity * self.delta_time.as_secs_f32() * self.player_controller.speed;

        let view_inverse = self.camera.transform.compute_matrix();

        let proj_matrix =
            glm::perspective(self.aspect_ratio(), glm::pi::<f32>() / 2.5, 0.1, 1000.0);

        let proj_inverse = glm::inverse(&proj_matrix);

        let uniform_buffer_data = GlobalUniforms {
            view_inverse,
            proj_inverse,
        };

        let uniforms_buffer = self.vk_controller.uniforms_buffer.as_mut().unwrap();

        unsafe {
            let buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .buffer(uniforms_buffer.buffer)
                .size(uniforms_buffer.size);

            self.vk_controller.device.cmd_pipeline_barrier(
                self.vk_controller.rt_command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::DEVICE_GROUP,
                &[],
                &[buffer_barrier],
                &[],
            );
        }

        unsafe {
            self.vk_controller.device.cmd_update_buffer(
                self.vk_controller.rt_command_buffer,
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

            self.vk_controller.device.cmd_pipeline_barrier(
                self.vk_controller.rt_command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::DEVICE_GROUP,
                &[],
                &[buffer_barrier],
                &[],
            );
        }
    }

    pub fn new(event_loop: &ActiveEventLoop, window_width: u32, window_height: u32) -> Self {
        let camera = CameraTransform {
            transform: Transform::from_xyz(0.0, 0.0, -2.0)
                .looking_at(bevy_math::Vec3::ZERO, bevy_math::Vec3::Y),
        };

        let mut vk_controller = VkController::new(event_loop, window_width, window_height);
        vk_controller.init();

        AppBase {
            vk_controller,
            current_frames_counter: 0,
            frame_start: std::time::Instant::now(),
            last_frame_update: std::time::Instant::now(),
            last_second: std::time::Instant::now(),
            delta_time: std::time::Duration::ZERO,
            player_controller: PlayerController::default(),
            camera,
            sensitivity: 0.001,
            resized: false,
            focused: false,
        }
    }

    pub fn main_loop(&mut self) {
        let rt_command_buffer = self.vk_controller.rt_command_buffer;
        unsafe {
            self.vk_controller
                .device
                .wait_for_fences(
                    &[self.vk_controller.swapchain_acquire_fence],
                    true,
                    std::u64::MAX,
                )
                .expect("Wait for fence failed.");

            self.vk_controller
                .device
                .reset_fences(&[self.vk_controller.swapchain_acquire_fence])
                .expect("Reset fences failed.");
        }

        if self.resized {
            self.vk_controller.recreate_swapchain().unwrap();
            self.resized = false;
        }

        self.current_frames_counter += 1;

        self.frame_start = std::time::Instant::now();

        self.update_delta_time();

        if self.frame_start.duration_since(self.last_second).as_secs() > 0 {
            // println!("{}fps", self.current_frames_counter);
            self.reset_fps_counter();
        }

        let (present_index, _) = unsafe {
            self.vk_controller.swapchain_loader.acquire_next_image(
                self.vk_controller.swapchain,
                std::u64::MAX,
                self.vk_controller.present_complete_semaphore,
                self.vk_controller.swapchain_acquire_fence,
            )
        }
        .unwrap();

        let current_swapchain_image = self.vk_controller.present_images[present_index as usize];

        unsafe {
            self.vk_controller
                .device
                .wait_for_fences(
                    &[self.vk_controller.draw_commands_reuse_fence],
                    true,
                    std::u64::MAX,
                )
                .expect("Wait for fence failed.");

            self.vk_controller
                .device
                .reset_fences(&[self.vk_controller.draw_commands_reuse_fence])
                .expect("Reset fences failed.");

            self.vk_controller
                .device
                .reset_command_buffer(
                    rt_command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Reset command buffer failed.");

            let rt_command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.vk_controller
                .device
                .begin_command_buffer(rt_command_buffer, &rt_command_buffer_begin_info)
                .expect("Begin commandbuffer");

            self.update_camera();

            // full rt pass
            {
                self.vk_controller.device.cmd_bind_pipeline(
                    rt_command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.vk_controller.pipeline.unwrap(),
                );
                self.vk_controller.device.cmd_bind_descriptor_sets(
                    rt_command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.vk_controller.pipeline_layout.unwrap(),
                    0,
                    &[
                        self.vk_controller.rt_descriptor_set.unwrap(),
                        self.vk_controller.uniforms_descriptor_set.unwrap(),
                    ],
                    &[],
                );
                self.vk_controller
                    .ray_tracing_pipeline_loader
                    .cmd_trace_rays(
                        rt_command_buffer,
                        &self.vk_controller.sbt_raygen_region.unwrap(),
                        &self.vk_controller.sbt_miss_region.unwrap(),
                        &self.vk_controller.sbt_hit_region.unwrap(),
                        &self.vk_controller.sbt_call_region.unwrap(),
                        self.vk_controller.surface_resolution.width,
                        self.vk_controller.surface_resolution.height,
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

                self.vk_controller.device.cmd_pipeline_barrier(
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
                    .image(self.vk_controller.rt_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                self.vk_controller.device.cmd_pipeline_barrier(
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
                            .width(self.vk_controller.surface_resolution.width)
                            .height(self.vk_controller.surface_resolution.height)
                            .depth(1),
                    );

                self.vk_controller.device.cmd_copy_image(
                    rt_command_buffer,
                    self.vk_controller.rt_image,
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

                self.vk_controller.device.cmd_pipeline_barrier(
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
                    .image(self.vk_controller.rt_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                self.vk_controller.device.cmd_pipeline_barrier(
                    rt_command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier],
                );
            }

            self.vk_controller
                .device
                .end_command_buffer(rt_command_buffer)
                .expect("End commandbuffer");

            let command_buffers = vec![rt_command_buffer];
            let wait_semaphores = &[self.vk_controller.present_complete_semaphore];
            let signal_semaphores = &[self.vk_controller.rendering_complete_semaphore];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&command_buffers)
                .signal_semaphores(signal_semaphores);

            self.vk_controller
                .device
                .queue_submit(
                    self.vk_controller.present_queue,
                    &[submit_info],
                    self.vk_controller.draw_commands_reuse_fence,
                )
                .expect("queue submit failed.");
        }

        let wait_semaphors = [self.vk_controller.rendering_complete_semaphore];
        let swapchains = [self.vk_controller.swapchain];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphors)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            if let Err(code) = self
                .vk_controller
                .swapchain_loader
                .queue_present(self.vk_controller.present_queue, &present_info)
            {
                match code {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.vk_controller.recreate_swapchain().unwrap()
                    }
                    _ => (),
                }
            }
        }
    }
}
