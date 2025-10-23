//! GPU Benchmark: Compare CPU vs GPU diffusion performance on Apple Silicon
//!
//! Run with: cargo run --example gpu_benchmark --features gpu --release
//! 
//! This example benchmarks diffusion on different grid sizes and compares:
//! - CPU performance (with Rayon parallelization)
//! - GPU performance (Metal backend on Apple Silicon via wgpu)

#[cfg(feature = "gpu")]
use ai_toroidal_memory::GpuDiffusionEngine;
use ai_toroidal_memory::{DiffusionEngine, DiffusionConfig, ToroidalMemory};
use std::time::Instant;

fn benchmark_cpu(width: usize, height: usize, steps: usize) -> f64 {
    println!("\n  CPU Benchmark: {}x{} grid, {} steps", width, height, steps);
    
    let mut memory = ToroidalMemory::new(width, height);
    
    // Initialize with random pattern
    for y in 0..height {
        for x in 0..width {
            let value = ((x + y) as f64 % 10.0) / 10.0;
            memory.set(x as isize, y as isize, value);
        }
    }
    
    let config = DiffusionConfig {
        diffusion_rate: 0.1,
        decay_rate: 0.05,
        threshold: 0.1,
    };
    
    let engine = DiffusionEngine::new(config);
    
    let start = Instant::now();
    for _ in 0..steps {
        memory = engine.step(&memory);
    }
    let elapsed = start.elapsed().as_secs_f64();
    
    println!("    ✓ Completed in {:.3}s ({:.1} steps/sec)", 
             elapsed, steps as f64 / elapsed);
    
    elapsed
}

#[cfg(feature = "gpu")]
async fn benchmark_gpu(width: usize, height: usize, steps: usize) -> f64 {
    println!("\n  GPU Benchmark: {}x{} grid, {} steps", width, height, steps);
    
    let mut gpu = match GpuDiffusionEngine::new(width, height, 0.1, 0.05, 0.1).await {
        Ok(g) => g,
        Err(e) => {
            println!("    ✗ GPU initialization failed: {}", e);
            return 0.0;
        }
    };
    
    // Initialize with random pattern
    let mut data = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            data[y * width + x] = ((x + y) as f32 % 10.0) / 10.0;
        }
    }
    
    gpu.upload_data(&data).await.ok();
    
    let start = Instant::now();
    for _ in 0..steps {
        gpu.diffusion_step().await.ok();
    }
    let elapsed = start.elapsed().as_secs_f64();
    
    println!("    ✓ Completed in {:.3}s ({:.1} steps/sec)", 
             elapsed, steps as f64 / elapsed);
    
    elapsed
}

#[cfg(not(feature = "gpu"))]
async fn benchmark_gpu(_width: usize, _height: usize, _steps: usize) -> f64 {
    println!("\n  GPU Benchmark: SKIPPED (compile with --features gpu)");
    0.0
}

#[tokio::main]
async fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         GPU vs CPU Diffusion Benchmark                     ║");
    println!("║         Apple Silicon Performance Comparison               ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    
    let test_cases = vec![
        (50, 50, 10),
        (100, 100, 10),
        (200, 200, 5),
        (500, 500, 2),
    ];
    
    println!("\n📊 Testing diffusion performance across grid sizes:\n");
    
    for (width, height, steps) in test_cases {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Grid Size: {}x{}", width, height);
        
        let cpu_time = benchmark_cpu(width, height, steps);
        let gpu_time = benchmark_gpu(width, height, steps).await;
        
        if gpu_time > 0.0 {
            let speedup = cpu_time / gpu_time;
            println!("  \n  Speedup: {:.2}x", speedup);
            
            if speedup > 1.0 {
                println!("  ✓ GPU is {:.1}% faster!", (speedup - 1.0) * 100.0);
            } else {
                println!("  ℹ GPU is {:.1}% slower (overhead for small grids)", 
                        (1.0 - speedup) * 100.0);
            }
        }
    }
    
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n📈 Summary:");
    println!("  • GPU shines on larger grids (200x200+)");
    println!("  • Small grids may show GPU overhead due to transfer cost");
    println!("  • Metal backend provides native Apple Silicon acceleration");
    println!("  • Use --features gpu to enable GPU support");
}
