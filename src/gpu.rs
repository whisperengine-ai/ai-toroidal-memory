/// GPU Acceleration for Toroidal Memory
/// 
/// This module provides GPU-accelerated diffusion using wgpu (Metal backend on Apple Silicon)
/// 
/// Feature: Requires the 'gpu' feature to be enabled
/// 
/// Usage:
/// ```
/// # #[cfg(feature = "gpu")]
/// # {
/// use ai_toroidal_memory::gpu::GpuDiffusionEngine;
/// 
/// let mut gpu = GpuDiffusionEngine::new(100, 100, 0.25, 0.1, 0.01).await;
/// gpu.activate(50, 50, 1.0).await;
/// gpu.diffusion_step().await;
/// let result = gpu.download_data().await;
/// # }
/// ```

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub diffusion_rate: f32,
    pub decay_rate: f32,
}

/// GPU-accelerated diffusion engine
pub struct GpuDiffusionEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    
    current_buffer: wgpu::Buffer,
    next_buffer: wgpu::Buffer,
    config_buffer: wgpu::Buffer,
    
    config: Config,
    data: Vec<f32>,
}

impl GpuDiffusionEngine {
    /// Create a new GPU diffusion engine
    pub async fn new(
        width: usize,
        height: usize,
        diffusion_rate: f32,
        decay_rate: f32,
        _threshold: f32,
    ) -> Result<Self, String> {
        // Initialize wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
            flags: wgpu::InstanceFlags::empty(),
        });

        // Request adapter with Metal backend on Apple Silicon
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;

        // Get device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to request device: {}", e))?;

        let config = Config {
            width: width as u32,
            height: height as u32,
            diffusion_rate,
            decay_rate,
        };

        let size = (width * height * std::mem::size_of::<f32>()) as u64;

        // Create buffers
        let current_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Current memory buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let next_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next memory buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config buffer"),
            contents: bytemuck::cast_slice(&[config]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create shader module
        let shader_code = include_str!("diffusion_shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Diffusion shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_code)),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Diffusion bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Diffusion pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Diffusion pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        let data = vec![0.0; width * height];

        Ok(GpuDiffusionEngine {
            device,
            queue,
            pipeline,
            bind_group_layout,
            current_buffer,
            next_buffer,
            config_buffer,
            config,
            data,
        })
    }

    /// Upload data to GPU
    pub async fn upload_data(&mut self, data: &[f32]) -> Result<(), String> {
        if data.len() != self.data.len() {
            return Err("Data size mismatch".to_string());
        }
        self.data.copy_from_slice(data);
        self.queue.write_buffer(&self.current_buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    /// Download data from GPU
    pub async fn download_data(&self) -> Result<Vec<f32>, String> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            size: (self.data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.current_buffer,
            0,
            &staging_buffer,
            0,
            (self.data.len() * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| "Failed to map buffer".to_string())?
            .map_err(|e| format!("Buffer mapping error: {}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Activate a cell with given strength
    pub fn activate(&mut self, x: isize, y: isize, strength: f32) {
        let width = self.config.width as isize;
        let height = self.config.height as isize;

        let wx = ((x % width + width) % width) as usize;
        let wy = ((y % height + height) % height) as usize;

        let idx = wy * self.config.width as usize + wx;
        self.data[idx] = (self.data[idx] + strength).min(1.0);

        self.queue.write_buffer(
            &self.current_buffer,
            (idx * std::mem::size_of::<f32>()) as u64,
            bytemuck::cast_slice(&[self.data[idx]]),
        );
    }

    /// Run one diffusion step on GPU
    pub async fn diffusion_step(&mut self) -> Result<(), String> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diffusion bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.current_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.next_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.config_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Diffusion encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Diffusion compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = (self.config.width + 15) / 16;
            let workgroups_y = (self.config.height + 15) / 16;
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy next to current
        encoder.copy_buffer_to_buffer(
            &self.next_buffer,
            0,
            &self.current_buffer,
            0,
            (self.data.len() * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// Run multiple diffusion steps
    pub async fn run(&mut self, steps: usize) -> Result<(), String> {
        for _ in 0..steps {
            self.diffusion_step().await?;
        }
        Ok(())
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.config.width as usize, self.config.height as usize)
    }
}
