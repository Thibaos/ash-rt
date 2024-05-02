use ash::{ext, khr, vk, Device, Entry, Instance};
use bevy_transform::components::Transform;
use bytemuck::{bytes_of, Pod, Zeroable};
use glm::Mat4;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{default::Default, ffi::CStr, ops::Drop, os::raw::c_char};

use winit::{event_loop::ActiveEventLoop, window::Window};

use crate::{
    player_controller::PlayerController,
    utils::{
        aligned_size, create_shader_module, find_memorytype_index, get_buffer_device_address,
        get_memory_type_index, pick_physical_device_and_queue_family_indices,
        record_submit_commandbuffer, vulkan_debug_callback, BufferResource,
    },
};

#[repr(C)]
#[derive(Clone, Debug, Copy, Pod, Zeroable)]
pub struct GlobalUniforms {
    pub view_inverse: bevy_math::Mat4, // Camera inverse view matrix
    pub proj_inverse: Mat4,            // Camera inverse projection matrix
}

#[repr(C)]
#[derive(Clone, Debug, Copy, Pod, Zeroable)]
pub struct Voxel {
    pub position: glm::Vec3,
    _pad: f32,
}

pub struct CameraTransform {
    pub transform: Transform,
}

macro_rules! destroy_buffer {
    ($($buffer_option: expr, $device: expr), *) => {
        $(
        let unwrap_result = $buffer_option.as_ref().unwrap();
        $device.destroy_buffer(unwrap_result.buffer, None);
        $device.free_memory(unwrap_result.memory, None);
    )*
    };
}

pub struct AppBase<'a> {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub surface_loader: khr::surface::Instance,
    pub swapchain_loader: khr::swapchain::Device,
    pub debug_utils_loader: ext::debug_utils::Instance,
    pub ray_tracing_pipeline_loader: khr::ray_tracing_pipeline::Device,
    pub acceleration_structure_loader: khr::acceleration_structure::Device,
    pub window: winit::window::Window,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,

    pub physical_device: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub present_queue: vk::Queue,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,

    desired_image_count: u32,
    pre_transform: vk::SurfaceTransformFlagsKHR,
    present_mode: vk::PresentModeKHR,

    pub swapchain: Option<vk::SwapchainKHR>,
    pub present_images: Option<Vec<vk::Image>>,
    pub present_image_views: Option<Vec<vk::ImageView>>,
    pub framebuffers: Option<Vec<vk::Framebuffer>>,

    pub pool: vk::CommandPool,
    pub draw_command_buffer: vk::CommandBuffer,
    pub setup_command_buffer: vk::CommandBuffer,
    pub update_command_buffer: vk::CommandBuffer,
    pub rt_command_buffer: vk::CommandBuffer,
    pub render_pass: Option<vk::RenderPass>,

    pub depth_image: Option<vk::Image>,
    pub depth_image_view: Option<vk::ImageView>,
    pub depth_image_memory: Option<vk::DeviceMemory>,

    pub rt_image: Option<vk::Image>,
    pub rt_image_view: Option<vk::ImageView>,
    pub rt_image_memory: Option<vk::DeviceMemory>,

    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,

    pub draw_commands_reuse_fence: vk::Fence,
    pub setup_commands_reuse_fence: vk::Fence,
    pub swapchain_acquire_fence: vk::Fence,

    pub resized: bool,
    pub focused: bool,

    pub as_geometry: Option<vk::AccelerationStructureGeometryKHR<'a>>,
    pub aabb_buffer: Option<BufferResource>,

    pub bottom_as: Option<vk::AccelerationStructureKHR>,
    pub bottom_as_buffer: Option<BufferResource>,

    pub top_as: Option<vk::AccelerationStructureKHR>,
    pub top_as_buffer: Option<BufferResource>,

    pub instance_count: Option<usize>,
    pub instance_buffer: Option<BufferResource>,

    pub colors_buffer: Option<BufferResource>,
    pub uniforms_buffer: Option<BufferResource>,
    pub voxels_buffer: Option<BufferResource>,

    pub rt_descriptor_pool: Option<vk::DescriptorPool>,
    pub rt_descriptor_set: Option<vk::DescriptorSet>,
    pub rt_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pub uniforms_descriptor_pool: Option<vk::DescriptorPool>,
    pub uniforms_descriptor_set: Option<vk::DescriptorSet>,
    pub uniforms_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pub pipeline: Option<vk::Pipeline>,
    pub pipeline_layout: Option<vk::PipelineLayout>,
    pub shader_group_count: Option<usize>,

    pub shader_binding_table_buffer: Option<BufferResource>,
    pub sbt_raygen_region: Option<vk::StridedDeviceAddressRegionKHR>,
    pub sbt_miss_region: Option<vk::StridedDeviceAddressRegionKHR>,
    pub sbt_hit_region: Option<vk::StridedDeviceAddressRegionKHR>,
    pub sbt_call_region: Option<vk::StridedDeviceAddressRegionKHR>,

    pub start: std::time::Instant,
    pub frame_start: std::time::Instant,
    pub current_frames_counter: u64,
    pub last_second: std::time::Instant,
    pub last_frame_update: std::time::Instant,
    pub delta_time: std::time::Duration,

    pub player_controller: PlayerController,
    pub camera: CameraTransform,
    pub sensitivity: f32,
}

impl AppBase<'_> {
    pub fn aspect_ratio(&self) -> f32 {
        self.surface_resolution.width as f32 / self.surface_resolution.height as f32
    }

    pub fn cleanup_swapchain(&self) -> anyhow::Result<()> {
        unsafe {
            self.device
                .destroy_image_view(self.rt_image_view.unwrap(), None);
            self.device.destroy_image(self.rt_image.unwrap(), None);
            self.device.free_memory(self.rt_image_memory.unwrap(), None);
        }

        for framebuffer in self.framebuffers.as_ref().unwrap() {
            unsafe { self.device.destroy_framebuffer(*framebuffer, None) };
        }

        unsafe {
            self.device
                .destroy_render_pass(self.render_pass.unwrap(), None)
        };

        for image_view in self.present_image_views.as_ref().unwrap() {
            unsafe { self.device.destroy_image_view(*image_view, None) };
        }

        unsafe { self.device.destroy_image(self.depth_image.unwrap(), None) };
        unsafe {
            self.device
                .destroy_image_view(self.depth_image_view.unwrap(), None)
        };
        unsafe {
            self.device
                .free_memory(self.depth_image_memory.unwrap(), None)
        };

        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain.unwrap(), None)
        };

        Ok(())
    }

    fn create_swapchain(&mut self) -> anyhow::Result<()> {
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(self.desired_image_count)
            .image_color_space(self.surface_format.color_space)
            .image_format(self.surface_format.format)
            .image_extent(self.surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(self.pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
        }?;

        self.swapchain = Some(swapchain);

        Ok(())
    }

    fn create_image_views(&mut self) -> anyhow::Result<()> {
        let present_images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(self.swapchain.unwrap())
        }
        .unwrap();
        let present_image_views: Vec<vk::ImageView> = present_images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                unsafe { self.device.create_image_view(&create_view_info, None) }.unwrap()
            })
            .collect();

        let depth_image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D16_UNORM)
            .extent(self.surface_resolution.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let depth_image = unsafe { self.device.create_image(&depth_image_create_info, None) }?;
        let depth_image_memory_req =
            unsafe { self.device.get_image_memory_requirements(depth_image) };
        let depth_image_memory_index = find_memorytype_index(
            &depth_image_memory_req,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("Unable to find suitable memory index for depth image.");

        let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(depth_image_memory_req.size)
            .memory_type_index(depth_image_memory_index);

        let depth_image_memory = unsafe {
            self.device
                .allocate_memory(&depth_image_allocate_info, None)
        }?;

        unsafe {
            self.device
                .bind_image_memory(depth_image, depth_image_memory, 0)
        }
        .expect("Unable to bind depth image memory");

        record_submit_commandbuffer(
            &self.device,
            self.setup_command_buffer,
            self.setup_commands_reuse_fence,
            self.present_queue,
            &[],
            &[],
            &[],
            |device, setup_command_buffer| {
                let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                    .image(depth_image)
                    .dst_access_mask(
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1)
                            .level_count(1),
                    );

                unsafe {
                    device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers],
                    )
                };
            },
        );

        let depth_image_view_info = vk::ImageViewCreateInfo::default()
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1),
            )
            .image(depth_image)
            .format(depth_image_create_info.format)
            .view_type(vk::ImageViewType::TYPE_2D);

        let depth_image_view =
            unsafe { self.device.create_image_view(&depth_image_view_info, None) }?;

        self.present_images = Some(present_images);
        self.present_image_views = Some(present_image_views);

        self.depth_image = Some(depth_image);
        self.depth_image_view = Some(depth_image_view);
        self.depth_image_memory = Some(depth_image_memory);

        Ok(())
    }

    fn create_framebuffers(&mut self) -> anyhow::Result<()> {
        let renderpass_attachments = [
            vk::AttachmentDescription {
                format: self.surface_format.format,
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

        let subpass = vk::SubpassDescription::default()
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        let renderpass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&renderpass_attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        let render_pass = unsafe {
            self.device
                .create_render_pass(&renderpass_create_info, None)
        }
        .unwrap();

        let framebuffers: Vec<vk::Framebuffer> = self
            .present_image_views
            .as_ref()
            .unwrap()
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments = [present_image_view, self.depth_image_view.unwrap()];
                let frame_buffer_create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&framebuffer_attachments)
                    .width(self.surface_resolution.width)
                    .height(self.surface_resolution.height)
                    .layers(1);

                unsafe {
                    self.device
                        .create_framebuffer(&frame_buffer_create_info, None)
                }
                .unwrap()
            })
            .collect();

        self.render_pass = Some(render_pass);
        self.framebuffers = Some(framebuffers);

        Ok(())
    }

    fn update_rt_image_descriptor_set(&mut self) {
        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(self.rt_image_view.unwrap())];

        let image_write = vk::WriteDescriptorSet::default()
            .dst_set(self.rt_descriptor_set.unwrap())
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info);

        unsafe {
            self.device.update_descriptor_sets(&[image_write], &[]);
        }
    }

    pub fn create_rt_image(&mut self) -> anyhow::Result<()> {
        let rt_image = {
            let image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(self.surface_format.format)
                .extent(
                    vk::Extent3D::default()
                        .width(self.surface_resolution.width)
                        .height(self.surface_resolution.height)
                        .depth(1),
                )
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                );

            unsafe { self.device.create_image(&image_create_info, None) }?
        };

        let rt_image_memory = {
            let mem_reqs = unsafe { self.device.get_image_memory_requirements(rt_image) };
            let mem_alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(get_memory_type_index(
                    self.device_memory_properties,
                    mem_reqs.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ));

            unsafe { self.device.allocate_memory(&mem_alloc_info, None) }?
        };

        unsafe { self.device.bind_image_memory(rt_image, rt_image_memory, 0) }?;

        let rt_image_view = {
            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.surface_format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(rt_image);

            unsafe { self.device.create_image_view(&image_view_create_info, None) }?
        };

        // rt image to general layout
        {
            let command_buffer = {
                let allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY);

                let command_buffers =
                    unsafe { self.device.allocate_command_buffers(&allocate_info) }.unwrap();
                command_buffers[0]
            };

            unsafe {
                self.device.begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
            }
            .unwrap();

            let image_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(rt_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier],
                );

                self.device.end_command_buffer(command_buffer).unwrap();
            }

            let command_buffers = [command_buffer];

            let submit_infos = [vk::SubmitInfo::default().command_buffers(&command_buffers)];

            unsafe {
                self.device
                    .queue_submit(self.graphics_queue, &submit_infos, vk::Fence::null())
                    .expect("Failed to execute queue submit.");

                self.device.queue_wait_idle(self.graphics_queue).unwrap();
                self.device
                    .free_command_buffers(self.pool, &[command_buffer]);
            }
        }

        self.rt_image = Some(rt_image);
        self.rt_image_view = Some(rt_image_view);
        self.rt_image_memory = Some(rt_image_memory);

        if self.rt_descriptor_set.is_some() {
            self.update_rt_image_descriptor_set();
        }

        Ok(())
    }

    pub fn recreate_swapchain(&mut self) -> anyhow::Result<()> {
        unsafe { self.device.device_wait_idle() }?;

        let size = self.window.inner_size();
        self.surface_resolution = vk::Extent2D {
            width: size.width,
            height: size.height,
        };

        println!(
            "Recreating swapchain with new resolution: ({}, {})",
            size.width, size.height
        );

        unsafe {
            self.device
                .destroy_image_view(self.rt_image_view.unwrap(), None);
            self.device.destroy_image(self.rt_image.unwrap(), None);
            self.device.free_memory(self.rt_image_memory.unwrap(), None);
        }

        for framebuffer in self.framebuffers.as_ref().unwrap() {
            unsafe { self.device.destroy_framebuffer(*framebuffer, None) };
        }

        unsafe {
            self.device
                .destroy_render_pass(self.render_pass.unwrap(), None)
        };

        for image_view in self.present_image_views.as_ref().unwrap() {
            unsafe { self.device.destroy_image_view(*image_view, None) };
        }

        unsafe { self.device.destroy_image(self.depth_image.unwrap(), None) };
        unsafe {
            self.device
                .destroy_image_view(self.depth_image_view.unwrap(), None)
        };
        unsafe {
            self.device
                .free_memory(self.depth_image_memory.unwrap(), None)
        };

        let surface_capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }
        .unwrap();

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let old_swapchain = self.swapchain.unwrap();

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(self.desired_image_count)
            .image_color_space(self.surface_format.color_space)
            .image_format(self.surface_format.format)
            .image_extent(self.surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            .image_array_layers(1)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
        }?;

        unsafe {
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }

        self.swapchain = Some(swapchain);

        self.create_image_views()?;
        self.create_framebuffers()?;
        self.create_rt_image()?;

        Ok(())
    }

    fn create_geometry(&mut self) {
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
                &self.device,
                self.device_memory_properties,
            );

            aabb_buffer.store(&corners, &self.device);

            aabb_buffer
        };

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default().data(
                    vk::DeviceOrHostAddressConstKHR {
                        device_address: unsafe {
                            get_buffer_device_address(&self.device, aabb_buffer.buffer)
                        },
                    },
                ),
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        self.as_geometry = Some(geometry);
        self.aabb_buffer = Some(aabb_buffer);
    }

    fn create_bottom_as(&mut self) {
        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(1)
            .primitive_offset(0)
            .transform_offset(0);

        let geometries = [self.as_geometry.unwrap()];

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

        unsafe {
            self.acceleration_structure_loader
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[1],
                    &mut size_info,
                )
        };

        let bottom_as_buffer = BufferResource::new(
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &self.device,
            self.device_memory_properties,
        );

        let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(build_info.ty)
            .size(size_info.acceleration_structure_size)
            .buffer(bottom_as_buffer.buffer)
            .offset(0);

        let bottom_as = unsafe {
            self.acceleration_structure_loader
                .create_acceleration_structure(&as_create_info, None)
        }
        .unwrap();

        build_info.dst_acceleration_structure = bottom_as;

        let scratch_buffer = BufferResource::new(
            size_info.build_scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &self.device,
            self.device_memory_properties,
        );

        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: unsafe {
                get_buffer_device_address(&self.device, scratch_buffer.buffer)
            },
        };

        let build_command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers =
                unsafe { self.device.allocate_command_buffers(&allocate_info) }.unwrap();
            command_buffers[0]
        };

        unsafe {
            self.device
                .begin_command_buffer(
                    build_command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            self.acceleration_structure_loader
                .cmd_build_acceleration_structures(
                    build_command_buffer,
                    &[build_info],
                    &[&[build_range_info]],
                );
            self.device
                .end_command_buffer(build_command_buffer)
                .unwrap();
            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[build_command_buffer])],
                    vk::Fence::null(),
                )
                .expect("queue submit failed.");

            self.device.queue_wait_idle(self.graphics_queue).unwrap();
            self.device
                .free_command_buffers(self.pool, &[build_command_buffer]);
            scratch_buffer.destroy(&self.device);
        }

        self.bottom_as = Some(bottom_as);
        self.bottom_as_buffer = Some(bottom_as_buffer);
    }

    fn create_as_instances(&mut self) {
        let accel_handle = {
            let as_addr_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(self.bottom_as.unwrap());
            unsafe {
                self.acceleration_structure_loader
                    .get_acceleration_structure_device_address(&as_addr_info)
            }
        };

        let transform_0: [f32; 12] = [1.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0];
        let transform_1: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0];
        let transform_2: [f32; 12] = [1.0, 0.0, 0.0, 1.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];

        let instances = vec![
            vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: transform_0,
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
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
                instance_custom_index_and_mask: vk::Packed24_8::new(1, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
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
                instance_custom_index_and_mask: vk::Packed24_8::new(2, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
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
            &self.device,
            self.device_memory_properties,
        );

        instance_buffer.store(&instances, &self.device);

        self.instance_count = Some(instances.len());
        self.instance_buffer = Some(instance_buffer);
    }

    fn create_top_as(&mut self) {
        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .first_vertex(0)
            .primitive_count(self.instance_count.unwrap() as u32)
            .primitive_offset(0)
            .transform_offset(0);

        let build_command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers =
                unsafe { self.device.allocate_command_buffers(&allocate_info) }.unwrap();
            command_buffers[0]
        };

        unsafe {
            self.device
                .begin_command_buffer(
                    build_command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR);
            self.device.cmd_pipeline_barrier(
                build_command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }

        let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: unsafe {
                    get_buffer_device_address(
                        &self.device,
                        self.instance_buffer.as_ref().unwrap().buffer,
                    )
                },
            });

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances });

        let geometries = [geometry];

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.acceleration_structure_loader
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[build_range_info.primitive_count],
                    &mut size_info,
                )
        }

        let top_as_buffer = BufferResource::new(
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &self.device,
            self.device_memory_properties,
        );

        let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(build_info.ty)
            .size(size_info.acceleration_structure_size)
            .buffer(top_as_buffer.buffer)
            .offset(0);

        let top_as = unsafe {
            self.acceleration_structure_loader
                .create_acceleration_structure(&as_create_info, None)
        }
        .unwrap();

        build_info.dst_acceleration_structure = top_as;

        let scratch_buffer = BufferResource::new(
            size_info.build_scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &self.device,
            self.device_memory_properties,
        );

        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: unsafe {
                get_buffer_device_address(&self.device, scratch_buffer.buffer)
            },
        };

        unsafe {
            self.acceleration_structure_loader
                .cmd_build_acceleration_structures(
                    build_command_buffer,
                    &[build_info],
                    &[&[build_range_info]],
                );
            self.device
                .end_command_buffer(build_command_buffer)
                .unwrap();
            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[build_command_buffer])],
                    vk::Fence::null(),
                )
                .expect("queue submit failed.");

            self.device.queue_wait_idle(self.graphics_queue).unwrap();
            self.device
                .free_command_buffers(self.pool, &[build_command_buffer]);
            scratch_buffer.destroy(&self.device);
        }

        self.top_as = Some(top_as);
        self.top_as_buffer = Some(top_as_buffer);
    }

    pub fn create_colors_buffer(&mut self) {
        let colors = [glm::Vec3::x(), glm::Vec3::y(), glm::Vec3::z()];
        let data = bytes_of(&colors);

        let mut colors_buffer = BufferResource::new(
            std::mem::size_of_val(&colors) as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.device,
            self.device_memory_properties,
        );
        colors_buffer.store(&data, &self.device);

        self.colors_buffer = Some(colors_buffer);
    }

    pub fn create_uniforms_buffer(&mut self) {
        let global_uniforms = GlobalUniforms::zeroed();

        let data = &[global_uniforms];

        let mut uniforms_buffer = BufferResource::new(
            std::mem::size_of_val(data) as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.device,
            self.device_memory_properties,
        );
        uniforms_buffer.store(data, &self.device);

        self.uniforms_buffer = Some(uniforms_buffer);
    }

    fn create_data_structures(&mut self) {
        self.create_geometry();
        self.create_bottom_as();
        self.create_as_instances();
        self.create_top_as();
        self.create_colors_buffer();

        let position0 = glm::Vec3::new(-1.5, 1.0, 10.0);
        let position1 = glm::Vec3::new(0.0, -1.0, 0.0);
        let position2 = glm::Vec3::new(1.5, 1.0, 0.0);

        let voxels = &[
            Voxel {
                position: position0,
                _pad: 0.0,
            },
            Voxel {
                position: position1,
                _pad: 0.0,
            },
            Voxel {
                position: position2,
                _pad: 0.0,
            },
        ];

        let mut voxels_buffer = BufferResource::new(
            std::mem::size_of_val(voxels) as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.device,
            self.device_memory_properties,
        );
        voxels_buffer.store(voxels, &self.device);

        self.voxels_buffer = Some(voxels_buffer);

        self.create_uniforms_buffer();
    }

    fn create_descriptor_sets(&mut self) -> anyhow::Result<()> {
        let mut rt_count_allocate_info =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(&[1]);

        let rt_descriptor_sizes = [
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
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
        ];

        let rt_descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&rt_descriptor_sizes)
            .max_sets(1);

        let rt_descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&rt_descriptor_pool_info, None)
        }
        .unwrap();

        let rt_binding_flags_inner = [
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
        ];

        let mut rt_binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
            .binding_flags(&rt_binding_flags_inner);

        let rt_descriptor_set_layout = unsafe {
            self.device.create_descriptor_set_layout(
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
                        vk::DescriptorSetLayoutBinding::default()
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::INTERSECTION_KHR)
                            .binding(3),
                    ])
                    .push_next(&mut rt_binding_flags),
                None,
            )
        }?;

        let uniforms_descriptor_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }];

        const RGEN_SHADER: &[u8] = include_bytes!("..\\shaders\\spv\\rgen.spv");
        const RCHIT_SHADER: &[u8] = include_bytes!("..\\shaders\\spv\\rchit.spv");
        const RMISS_SHADER: &[u8] = include_bytes!("..\\shaders\\spv\\rmiss.spv");
        const RINT_SHADER: &[u8] = include_bytes!("..\\shaders\\spv\\rint.spv");

        let rgen_module = unsafe { create_shader_module(&self.device, RGEN_SHADER) }?;
        let rchit_module = unsafe { create_shader_module(&self.device, RCHIT_SHADER) }?;
        let rmiss_module = unsafe { create_shader_module(&self.device, RMISS_SHADER) }?;
        let rint_module = unsafe { create_shader_module(&self.device, RINT_SHADER) }?;

        let uniforms_binding_flags_inner = [vk::DescriptorBindingFlagsEXT::empty()];

        let mut uniforms_binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
                .binding_flags(&uniforms_binding_flags_inner);

        let uniforms_descriptor_set_layout = unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&[vk::DescriptorSetLayoutBinding::default()
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                        .binding(0)])
                    .push_next(&mut uniforms_binding_flags),
                None,
            )
        }?;

        let uniforms_descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&uniforms_descriptor_sizes)
            .max_sets(1);

        let uniforms_descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&uniforms_descriptor_pool_info, None)
        }
        .unwrap();

        let layouts = vec![rt_descriptor_set_layout, uniforms_descriptor_set_layout];
        let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_create_info, None)
        }?;

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
                .module(rgen_module)
                .name(std::ffi::CStr::from_bytes_with_nul(b"main\0")?),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(rchit_module)
                .name(std::ffi::CStr::from_bytes_with_nul(b"main\0")?),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
                .module(rint_module)
                .name(std::ffi::CStr::from_bytes_with_nul(b"main\0")?),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(rmiss_module)
                .name(std::ffi::CStr::from_bytes_with_nul(b"main\0")?),
        ];

        let pipeline = unsafe {
            self.ray_tracing_pipeline_loader
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
            self.device.destroy_shader_module(rgen_module, None);
            self.device.destroy_shader_module(rchit_module, None);
            self.device.destroy_shader_module(rmiss_module, None);
            self.device.destroy_shader_module(rint_module, None);
        }

        let rt_descriptor_sets = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(rt_descriptor_pool)
                    .set_layouts(&[rt_descriptor_set_layout])
                    .push_next(&mut rt_count_allocate_info),
            )
        }
        .unwrap();

        let rt_descriptor_set = rt_descriptor_sets[0];

        let mut uniforms_count_allocate_info =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(&[1]);

        let uniforms_descriptor_sets = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(uniforms_descriptor_pool)
                    .set_layouts(&[uniforms_descriptor_set_layout])
                    .push_next(&mut uniforms_count_allocate_info),
            )
        }
        .unwrap();

        let uniforms_descriptor_set = uniforms_descriptor_sets[0];

        let accel_structs = [self.top_as.unwrap()];

        let mut accel_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&accel_structs);

        let accel_write = vk::WriteDescriptorSet::default()
            .dst_set(rt_descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .push_next(&mut accel_info);

        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(self.rt_image_view.unwrap())];

        let image_write = vk::WriteDescriptorSet::default()
            .dst_set(rt_descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info);

        let colors_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.colors_buffer.as_ref().unwrap().buffer)
            .range(vk::WHOLE_SIZE)];

        let colors_buffer_write = vk::WriteDescriptorSet::default()
            .dst_set(rt_descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&colors_buffer_info);

        let voxels_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.voxels_buffer.as_ref().unwrap().buffer)
            .range(vk::WHOLE_SIZE)];

        let voxels_buffer_write = vk::WriteDescriptorSet::default()
            .dst_set(rt_descriptor_set)
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&voxels_buffer_info);

        unsafe {
            self.device.update_descriptor_sets(
                &[
                    accel_write,
                    image_write,
                    colors_buffer_write,
                    voxels_buffer_write,
                ],
                &[],
            );
        }

        let uniforms_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.uniforms_buffer.as_ref().unwrap().buffer)
            .range(vk::WHOLE_SIZE)];

        let uniforms_buffer_write = vk::WriteDescriptorSet::default()
            .dst_set(uniforms_descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&uniforms_buffer_info);

        unsafe {
            self.device
                .update_descriptor_sets(&[uniforms_buffer_write], &[]);
        }

        self.rt_descriptor_pool = Some(rt_descriptor_pool);
        self.rt_descriptor_set = Some(rt_descriptor_set);
        self.rt_descriptor_set_layout = Some(rt_descriptor_set_layout);

        self.uniforms_descriptor_pool = Some(uniforms_descriptor_pool);
        self.uniforms_descriptor_set = Some(uniforms_descriptor_set);
        self.uniforms_descriptor_set_layout = Some(uniforms_descriptor_set_layout);

        self.pipeline = Some(pipeline);
        self.pipeline_layout = Some(pipeline_layout);
        self.shader_group_count = Some(shader_groups.len());

        Ok(())
    }

    fn create_rt_sbt(&mut self) -> anyhow::Result<()> {
        let mut rt_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();

        {
            let mut physical_device_properties2 =
                vk::PhysicalDeviceProperties2::default().push_next(&mut rt_pipeline_properties);

            unsafe {
                self.instance.get_physical_device_properties2(
                    self.physical_device,
                    &mut physical_device_properties2,
                );
            }
        }

        let handle_size_aligned = aligned_size(
            rt_pipeline_properties.shader_group_handle_size,
            rt_pipeline_properties.shader_group_base_alignment,
        ) as u64;

        let shader_binding_table_buffer = {
            let incoming_table_data = unsafe {
                self.ray_tracing_pipeline_loader
                    .get_ray_tracing_shader_group_handles(
                        self.pipeline.unwrap(),
                        0,
                        self.shader_group_count.unwrap() as u32,
                        self.shader_group_count.unwrap()
                            * rt_pipeline_properties.shader_group_handle_size as usize,
                    )
            }?;

            let table_size = self.shader_group_count.unwrap() * handle_size_aligned as usize;
            let mut table_data = vec![0u8; table_size];

            for i in 0..self.shader_group_count.unwrap() {
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
                &self.device,
                self.device_memory_properties,
            );

            shader_binding_table_buffer.store(&table_data, &self.device);

            shader_binding_table_buffer
        };

        // |[ raygen shader ]|[ hit shader  ]|[ miss shader ]|
        // |                 |               |               |
        // | 0               | 1             | 2             | 3

        let sbt_address =
            unsafe { get_buffer_device_address(&self.device, shader_binding_table_buffer.buffer) };

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

        self.shader_binding_table_buffer = Some(shader_binding_table_buffer);
        self.sbt_raygen_region = Some(sbt_raygen_region);
        self.sbt_miss_region = Some(sbt_miss_region);
        self.sbt_hit_region = Some(sbt_hit_region);
        self.sbt_call_region = Some(sbt_call_region);

        Ok(())
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

            let window_scale = self
                .surface_resolution
                .height
                .min(self.surface_resolution.width) as f32;

            yaw -= delta.0 as f32 * self.sensitivity * window_scale * self.delta_time.as_secs_f32();
            pitch -=
                delta.1 as f32 * self.sensitivity * window_scale * self.delta_time.as_secs_f32();

            pitch = pitch.clamp(-glm::half_pi::<f32>(), glm::half_pi());

            self.camera.transform.rotation =
                bevy_math::Quat::from_axis_angle(bevy_math::Vec3::Y, yaw)
                    * bevy_math::Quat::from_axis_angle(bevy_math::Vec3::X, pitch);
        }
    }

    pub fn toggle_capture_mouse(&mut self) {
        if self.focused {
            self.focused = false;
            self.window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
            self.window.set_cursor_visible(true);
        } else {
            self.focused = true;
            self.window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .unwrap();
            self.window.set_cursor_visible(false);
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

        let uniforms_buffer = self.uniforms_buffer.as_mut().unwrap();

        unsafe {
            let buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .buffer(uniforms_buffer.buffer)
                .size(uniforms_buffer.size);

            self.device.cmd_pipeline_barrier(
                self.rt_command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::DEVICE_GROUP,
                &[],
                &[buffer_barrier],
                &[],
            );
        }

        unsafe {
            self.device.cmd_update_buffer(
                self.rt_command_buffer,
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

            self.device.cmd_pipeline_barrier(
                self.rt_command_buffer,
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
        let window_attributes = Window::default_attributes()
            .with_title("RT")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                f64::from(window_width),
                f64::from(window_height),
            ));

        let window = event_loop.create_window(window_attributes).unwrap();

        let entry = Entry::linked();
        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"VulkanRT\0") };

        let layer_names = unsafe {
            [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        };
        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap()
                .to_vec();
        extension_names.push(ext::debug_utils::NAME.as_ptr());

        let appinfo = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(app_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_2);

        let create_flags = vk::InstanceCreateFlags::default();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);

        let instance: Instance =
            unsafe { entry.create_instance(&create_info, None) }.expect("Instance creation error");

        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_utils_loader = ext::debug_utils::Instance::new(&entry, &instance);
        let debug_call_back =
            unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None) }.unwrap();
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
        }
        .unwrap();

        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let (physical_device, queue_family_index) = pick_physical_device_and_queue_family_indices(
            &instance,
            surface,
            &surface_loader,
            &[
                khr::acceleration_structure::NAME,
                khr::deferred_host_operations::NAME,
                khr::ray_tracing_pipeline::NAME,
            ],
        )
        .unwrap()
        .unwrap();

        let device: Device = {
            let priorities = [1.0];

            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);

            let mut features2 = vk::PhysicalDeviceFeatures2::default();
            unsafe {
                (instance.fp_v1_1().get_physical_device_features2)(physical_device, &mut features2)
            };

            let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
                .scalar_block_layout(true)
                .buffer_device_address(true)
                .vulkan_memory_model(true);

            let mut as_feature = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);

            let mut raytracing_pipeline =
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                    .ray_tracing_pipeline(true);

            let enabled_extension_names = [
                khr::swapchain::NAME.as_ptr(),
                khr::ray_tracing_pipeline::NAME.as_ptr(),
                khr::acceleration_structure::NAME.as_ptr(),
                khr::deferred_host_operations::NAME.as_ptr(),
                vk::KHR_SPIRV_1_4_NAME.as_ptr(),
                vk::EXT_SCALAR_BLOCK_LAYOUT_NAME.as_ptr(),
                vk::KHR_GET_MEMORY_REQUIREMENTS2_NAME.as_ptr(),
            ];

            let queue_create_infos = [queue_create_info];

            let device_create_info = vk::DeviceCreateInfo::default()
                .push_next(&mut features2)
                .push_next(&mut features12)
                .push_next(&mut as_feature)
                .push_next(&mut raytracing_pipeline)
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&enabled_extension_names);

            unsafe { instance.create_device(physical_device, &device_create_info, None) }
                .expect("Failed to create logical Device!")
        };

        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let surface_format =
            unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
                .unwrap()[0];

        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .unwrap();

        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D {
                width: window_width,
                height: window_height,
            },
            _ => surface_capabilities.current_extent,
        };
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
        }
        .unwrap();
        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let pool = unsafe { device.create_command_pool(&pool_create_info, None) }.unwrap();

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(3)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }.unwrap();
        let setup_command_buffer = command_buffers[0];
        let update_command_buffer = command_buffers[1];
        let draw_command_buffer = command_buffers[2];

        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let draw_commands_reuse_fence =
            unsafe { device.create_fence(&fence_create_info, None) }.expect("Create fence failed.");
        let setup_commands_reuse_fence =
            unsafe { device.create_fence(&fence_create_info, None) }.expect("Create fence failed.");
        let swapchain_acquire_fence =
            unsafe { device.create_fence(&fence_create_info, None) }.expect("Create fence failed.");

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let present_complete_semaphore =
            unsafe { device.create_semaphore(&semaphore_create_info, None) }.unwrap();
        let rendering_complete_semaphore =
            unsafe { device.create_semaphore(&semaphore_create_info, None) }.unwrap();

        let ray_tracing_pipeline_loader =
            khr::ray_tracing_pipeline::Device::new(&instance, &device);

        let acceleration_structure_loader =
            khr::acceleration_structure::Device::new(&instance, &device);

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let camera = CameraTransform {
            transform: Transform::from_xyz(0.0, 0.0, -2.0)
                .looking_at(bevy_math::Vec3::ZERO, bevy_math::Vec3::Y),
        };

        let rt_command_buffer = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
                .expect("Failed to allocate Command Buffers!")[0]
        };

        AppBase {
            entry,
            ray_tracing_pipeline_loader,
            acceleration_structure_loader,
            instance,
            device,
            graphics_queue,
            queue_family_index,
            physical_device,
            device_memory_properties,
            window,
            surface_loader,
            surface_format,
            present_queue,
            surface_resolution,
            swapchain_loader,
            pool,
            draw_command_buffer,
            setup_command_buffer,
            update_command_buffer,
            rt_command_buffer,
            present_complete_semaphore,
            rendering_complete_semaphore,
            draw_commands_reuse_fence,
            setup_commands_reuse_fence,
            swapchain_acquire_fence,
            surface,
            debug_call_back,
            debug_utils_loader,
            desired_image_count,
            pre_transform,
            present_mode,
            current_frames_counter: 0,
            start: std::time::Instant::now(),
            frame_start: std::time::Instant::now(),
            last_frame_update: std::time::Instant::now(),
            last_second: std::time::Instant::now(),
            delta_time: std::time::Duration::ZERO,
            player_controller: PlayerController::default(),
            camera,
            sensitivity: 0.001,
            swapchain: None,
            present_images: None,
            present_image_views: None,
            depth_image: None,
            depth_image_view: None,
            depth_image_memory: None,
            rt_image: None,
            rt_image_view: None,
            rt_image_memory: None,
            render_pass: None,
            framebuffers: None,
            resized: false,
            focused: false,
            as_geometry: None,
            aabb_buffer: None,
            bottom_as: None,
            bottom_as_buffer: None,
            top_as: None,
            top_as_buffer: None,
            instance_count: None,
            instance_buffer: None,
            colors_buffer: None,
            uniforms_buffer: None,
            voxels_buffer: None,
            rt_descriptor_pool: None,
            rt_descriptor_set: None,
            rt_descriptor_set_layout: None,
            uniforms_descriptor_pool: None,
            uniforms_descriptor_set: None,
            uniforms_descriptor_set_layout: None,
            pipeline: None,
            pipeline_layout: None,
            shader_group_count: None,
            shader_binding_table_buffer: None,
            sbt_raygen_region: None,
            sbt_hit_region: None,
            sbt_miss_region: None,
            sbt_call_region: None,
        }
    }

    pub fn init(&mut self) {
        self.create_swapchain().unwrap();
        self.create_image_views().unwrap();
        self.create_framebuffers().unwrap();
        self.create_rt_image().unwrap();
        self.create_data_structures();
        self.create_descriptor_sets().unwrap();
        self.create_rt_sbt().unwrap();
    }

    pub fn main_loop(&mut self) {
        let rt_command_buffer = self.rt_command_buffer;
        unsafe {
            self.device
                .wait_for_fences(&[self.swapchain_acquire_fence], true, std::u64::MAX)
                .expect("Wait for fence failed.");

            self.device
                .reset_fences(&[self.swapchain_acquire_fence])
                .expect("Reset fences failed.");
        }

        if self.resized {
            self.recreate_swapchain().unwrap();
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
            self.swapchain_loader.acquire_next_image(
                self.swapchain.unwrap(),
                std::u64::MAX,
                self.present_complete_semaphore,
                self.swapchain_acquire_fence,
            )
        }
        .unwrap();

        let current_swapchain_image = self.present_images.as_ref().unwrap()[present_index as usize];

        unsafe {
            self.device
                .wait_for_fences(&[self.draw_commands_reuse_fence], true, std::u64::MAX)
                .expect("Wait for fence failed.");

            self.device
                .reset_fences(&[self.draw_commands_reuse_fence])
                .expect("Reset fences failed.");

            self.device
                .reset_command_buffer(
                    rt_command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Reset command buffer failed.");

            let rt_command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(rt_command_buffer, &rt_command_buffer_begin_info)
                .expect("Begin commandbuffer");

            self.update_camera();

            // full rt pass
            {
                self.device.cmd_bind_pipeline(
                    rt_command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline.unwrap(),
                );
                self.device.cmd_bind_descriptor_sets(
                    rt_command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline_layout.unwrap(),
                    0,
                    &[
                        self.rt_descriptor_set.unwrap(),
                        self.uniforms_descriptor_set.unwrap(),
                    ],
                    &[],
                );
                self.ray_tracing_pipeline_loader.cmd_trace_rays(
                    rt_command_buffer,
                    &self.sbt_raygen_region.unwrap(),
                    &self.sbt_miss_region.unwrap(),
                    &self.sbt_hit_region.unwrap(),
                    &self.sbt_call_region.unwrap(),
                    self.surface_resolution.width,
                    self.surface_resolution.height,
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

                self.device.cmd_pipeline_barrier(
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
                    .image(self.rt_image.unwrap())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                self.device.cmd_pipeline_barrier(
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
                            .width(self.surface_resolution.width)
                            .height(self.surface_resolution.height)
                            .depth(1),
                    );

                self.device.cmd_copy_image(
                    rt_command_buffer,
                    self.rt_image.unwrap(),
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

                self.device.cmd_pipeline_barrier(
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
                    .image(self.rt_image.unwrap())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                self.device.cmd_pipeline_barrier(
                    rt_command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier],
                );
            }

            self.device
                .end_command_buffer(rt_command_buffer)
                .expect("End commandbuffer");

            let command_buffers = vec![rt_command_buffer];
            let wait_semaphores = &[self.present_complete_semaphore];
            let signal_semaphores = &[self.rendering_complete_semaphore];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&command_buffers)
                .signal_semaphores(signal_semaphores);

            self.device
                .queue_submit(
                    self.present_queue,
                    &[submit_info],
                    self.draw_commands_reuse_fence,
                )
                .expect("queue submit failed.");
        }

        let wait_semaphors = [self.rendering_complete_semaphore];
        let swapchains = [self.swapchain.unwrap()];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphors)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            if let Err(code) = self
                .swapchain_loader
                .queue_present(self.present_queue, &present_info)
            {
                match code {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => self.recreate_swapchain().unwrap(),
                    _ => (),
                }
            }
        }
    }
}

impl Drop for AppBase<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.swapchain_acquire_fence], true, std::u64::MAX)
                .unwrap();

            self.device.device_wait_idle().unwrap();
            self.device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.device
                .destroy_semaphore(self.rendering_complete_semaphore, None);
            self.device
                .destroy_fence(self.draw_commands_reuse_fence, None);
            self.device
                .destroy_fence(self.setup_commands_reuse_fence, None);
            self.device
                .destroy_fence(self.swapchain_acquire_fence, None);

            self.cleanup_swapchain().unwrap();

            self.device
                .destroy_descriptor_pool(self.rt_descriptor_pool.unwrap(), None);
            self.device
                .destroy_descriptor_pool(self.uniforms_descriptor_pool.unwrap(), None);
            self.device.destroy_pipeline(self.pipeline.unwrap(), None);
            self.device
                .destroy_descriptor_set_layout(self.rt_descriptor_set_layout.unwrap(), None);
            self.device
                .destroy_descriptor_set_layout(self.uniforms_descriptor_set_layout.unwrap(), None);

            self.device
                .destroy_pipeline_layout(self.pipeline_layout.unwrap(), None);

            self.acceleration_structure_loader
                .destroy_acceleration_structure(self.bottom_as.unwrap(), None);

            self.acceleration_structure_loader
                .destroy_acceleration_structure(self.top_as.unwrap(), None);

            destroy_buffer!(self.bottom_as_buffer, self.device);
            destroy_buffer!(self.top_as_buffer, self.device);
            destroy_buffer!(self.colors_buffer, self.device);
            destroy_buffer!(self.instance_buffer, self.device);
            destroy_buffer!(self.aabb_buffer, self.device);
            destroy_buffer!(self.shader_binding_table_buffer, self.device);
            destroy_buffer!(self.uniforms_buffer, self.device);
            destroy_buffer!(self.voxels_buffer, self.device);

            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}
