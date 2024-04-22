use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};
use vulkano::{
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    VulkanLibrary,
};

struct App {}

impl App {
    fn new() -> Self {
        let event_loop = winit::event_loop::EventLoop::new().unwrap();

        let library = VulkanLibrary::new().unwrap();

        let mut required_extensions = InstanceExtensions {
            khr_surface: true,
            ..InstanceExtensions::empty()
        };
        match event_loop.display_handle().unwrap().as_raw() {
            RawDisplayHandle::Android(_) => required_extensions.khr_android_surface = true,
            // FIXME: `mvk_macos_surface` and `mvk_ios_surface` are deprecated.
            RawDisplayHandle::AppKit(_) => required_extensions.mvk_macos_surface = true,
            RawDisplayHandle::UiKit(_) => required_extensions.mvk_ios_surface = true,
            RawDisplayHandle::Windows(_) => required_extensions.khr_win32_surface = true,
            RawDisplayHandle::Wayland(_) => required_extensions.khr_wayland_surface = true,
            RawDisplayHandle::Xcb(_) => required_extensions.khr_xcb_surface = true,
            RawDisplayHandle::Xlib(_) => required_extensions.khr_xlib_surface = true,
            _ => unimplemented!(),
        }

        // Choose device extensions that we're going to use. In order to present images to a surface,
        // we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_acceleration_structure: true,
            khr_ray_tracing_pipeline: true,
            ..DeviceExtensions::empty()
        };

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .unwrap();

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                // The Vulkan specs guarantee that a compliant implementation must provide at least one
                // queue that supports compute operations.
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        mod shaders {
            vulkano_shaders::shader! {
                shaders: {
                    raygen: { ty: "raygen", path: "shaders/rt.rgen" },
                    chit: { ty: "closesthit", path: "shaders/rt.rchit" },
                    miss: { ty: "miss", path: "shaders/rt.rmiss" },
                    intersection: { ty: "intersection", path: "shaders/rt.rint" }
                },
                vulkan_version: "1.2",
                spirv_version: "1.6"
            }
        }

        App {}
    }
}
