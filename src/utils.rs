use std::{
    borrow::Cow,
    collections::HashSet,
    ffi::{c_char, CStr},
    ptr,
};

use ash::{khr, prelude::VkResult, util, vk, Device, Instance};

pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}

/// Helper function for submitting command buffers. Immediately waits for the fence before the command buffer
/// is executed. That way we can delay the waiting for the fences by 1 frame which is good for performance.
/// Make sure to create the fence in a signaled state on the first use.
#[allow(clippy::too_many_arguments)]
pub fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    base_pass: F,
) {
    unsafe {
        device
            .wait_for_fences(&[command_buffer_reuse_fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");

        device
            .reset_fences(&[command_buffer_reuse_fence])
            .expect("Reset fences failed.");

        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");

        base_pass(device, command_buffer);

        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(submit_queue, &[submit_info], command_buffer_reuse_fence)
            .expect("queue submit failed.");
    }
}

pub unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    vk::FALSE
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}

pub fn pick_physical_device_and_queue_family_indices(
    instance: &Instance,
    surface: vk::SurfaceKHR,
    surface_loader: &khr::surface::Instance,
    extensions: &[&CStr],
) -> VkResult<Option<(vk::PhysicalDevice, u32)>> {
    Ok(unsafe { instance.enumerate_physical_devices() }?
        .into_iter()
        .find_map(|physical_device| {
            let has_all_extesions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }.map(
                    |exts| {
                        let set: HashSet<&CStr> = exts
                            .iter()
                            .map(|ext| unsafe {
                                CStr::from_ptr(&ext.extension_name as *const c_char)
                            })
                            .collect();

                        extensions.iter().all(|ext| set.contains(ext))
                    },
                );
            if has_all_extesions != Ok(true) {
                return None;
            }

            let graphics_family =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                    .into_iter()
                    .enumerate()
                    .find(|(index, device_properties)| {
                        device_properties.queue_count > 0
                            && device_properties
                                .queue_flags
                                .contains(vk::QueueFlags::GRAPHICS)
                            && unsafe {
                                surface_loader.get_physical_device_surface_support(
                                    physical_device,
                                    *index as u32,
                                    surface,
                                )
                            }
                            .unwrap()
                    });

            graphics_family.map(|(i, _)| (physical_device, i as u32))
        }))
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
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
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
            let mut mapped_slice =
                util::Align::new(mapped_ptr, std::mem::align_of::<T>() as u64, size);
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

#[allow(unused)]
fn lerp(start: bevy_math::Vec3, end: bevy_math::Vec3, delta: f32) -> bevy_math::Vec3 {
    start + (end - start) * delta
}

pub unsafe fn create_shader_module(
    device: &ash::Device,
    code: &[u8],
) -> VkResult<vk::ShaderModule> {
    let shader_module_create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ShaderModuleCreateFlags::empty(),
        code_size: code.len(),
        p_code: code.as_ptr() as *const u32,
        ..Default::default()
    };

    device.create_shader_module(&shader_module_create_info, None)
}

pub fn aligned_size(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

pub unsafe fn get_buffer_device_address(device: &ash::Device, buffer: vk::Buffer) -> u64 {
    let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);

    device.get_buffer_device_address(&buffer_device_address_info)
}
