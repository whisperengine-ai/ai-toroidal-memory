/// WGSL Shader for Toroidal Memory Diffusion
/// 
/// This shader implements the diffusion algorithm on GPU
/// Each thread processes one cell of the toroidal memory

struct Config {
    width: u32,
    height: u32,
    diffusion_rate: f32,
    decay_rate: f32,
}

@group(0) @binding(0)
var<storage, read> current: array<f32>;

@group(0) @binding(1)
var<storage, read_write> next: array<f32>;

@group(0) @binding(2)
var<uniform> config: Config;

fn wrap(val: i32, size: i32) -> i32 {
    return ((val % size) + size) % size;
}

fn get_index(x: i32, y: i32) -> u32 {
    let wx = wrap(x, i32(config.width));
    let wy = wrap(y, i32(config.height));
    return u32(wy) * config.width + u32(wx);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    
    if (u32(x) >= config.width || u32(y) >= config.height) {
        return;
    }
    
    let idx = get_index(x, y);
    let current_val = current[idx];
    
    // Get toroidal neighbors (von Neumann neighborhood - 4-connected)
    let top_idx = get_index(x, y - 1);
    let bottom_idx = get_index(x, y + 1);
    let left_idx = get_index(x - 1, y);
    let right_idx = get_index(x + 1, y);
    
    let neighbors_sum = current[top_idx] + current[bottom_idx] + 
                        current[left_idx] + current[right_idx];
    let neighbors_avg = neighbors_sum / 4.0;
    
    // Diffusion equation:
    // new_value = current + diffusion_in - diffusion_out - decay
    let diffusion_in = neighbors_avg * config.diffusion_rate;
    let diffusion_out = current_val * config.diffusion_rate;
    let decay = current_val * config.decay_rate;
    
    let new_value = current_val + diffusion_in - diffusion_out - decay;
    
    // Clamp to [0, 1]
    next[idx] = clamp(new_value, 0.0, 1.0);
}
