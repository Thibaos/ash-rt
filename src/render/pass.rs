use std::sync::Arc;

use ash::Device;

use crate::utils::BufferResource;

pub struct Pass {
    pub device: Arc<Device>,

    buffers: Vec<BufferResource>,
}

impl Pass {
    fn new(device: Arc<Device>) -> Self {
        let buffers = vec![];

        Pass { device, buffers }
    }
}
