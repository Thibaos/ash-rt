use std::sync::Arc;

use ash::Device;

use crate::utils::BufferResource;

pub enum PipelineType {
    GraphicsPipeline,
    RayTracingPipeline,
}

pub struct Pass {
    pub device: Arc<Device>,
    pub pipeline_type: PipelineType,
    buffers: Vec<BufferResource>,
}

impl Pass {
    fn new(device: Arc<Device>, pipeline_type: PipelineType) -> Self {
        Pass {
            device,
            pipeline_type,
            buffers: vec![],
        }
    }
}
