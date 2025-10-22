/// Performance Benchmarking for Toroidal Memory
/// 
/// Demonstrates the performance improvements from parallel processing.
/// 
/// Run with: cargo run --example benchmark --release
/// Run without parallel: cargo run --example benchmark --release --no-default-features

use ai_toroidal_memory::{ToroidalMemory, DiffusionEngine, DiffusionConfig};
use std::time::Instant;

fn benchmark_diffusion(width: usize, height: usize, steps: usize) -> f64 {
    let mut memory = ToroidalMemory::new(width, height);
    
    // Initialize with some activation
    DiffusionEngine::activate_radius(&mut memory, 
        (width / 2) as isize, 
        (height / 2) as isize, 
        10, 
        1.0
    );
    
    let engine = DiffusionEngine::new(DiffusionConfig {
        diffusion_rate: 0.25,
        decay_rate: 0.1,
        threshold: 0.01,
    });
    
    let start = Instant::now();
    engine.run(&mut memory, steps);
    let elapsed = start.elapsed();
    
    elapsed.as_secs_f64()
}

fn benchmark_activation(width: usize, height: usize, radius: isize) -> f64 {
    let mut memory = ToroidalMemory::new(width, height);
    
    let start = Instant::now();
    DiffusionEngine::activate_radius(&mut memory, 
        (width / 2) as isize, 
        (height / 2) as isize, 
        radius, 
        1.0
    );
    let elapsed = start.elapsed();
    
    elapsed.as_secs_f64()
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     Toroidal Memory Performance Benchmarks                ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    
    #[cfg(feature = "parallel")]
    println!("✅ Parallel processing: ENABLED");
    #[cfg(not(feature = "parallel"))]
    println!("❌ Parallel processing: DISABLED");
    
    println!();
    println!("Available CPU cores: {}", num_cpus::get());
    println!();
    
    // Benchmark diffusion at different sizes
    println!("📊 Diffusion Benchmarks (10 steps)");
    println!("═══════════════════════════════════════════════════════════");
    
    let sizes = vec![
        (10, 10, "Small (10×10)"),
        (50, 50, "Medium (50×50)"),
        (100, 100, "Large (100×100)"),
        (200, 200, "Very Large (200×200)"),
        (500, 500, "Huge (500×500)"),
    ];
    
    for (width, height, label) in sizes {
        print!("{:20} ", label);
        let time = benchmark_diffusion(width, height, 10);
        let cells = width * height;
        let cells_per_sec = (cells as f64 * 10.0) / time;
        println!("{:8.3}s  ({:.0} cells/sec)", time, cells_per_sec);
    }
    
    println!();
    println!("📊 Activation Radius Benchmarks");
    println!("═══════════════════════════════════════════════════════════");
    
    let grid_size = 100;
    let radii = vec![5, 10, 20, 30, 50];
    
    for radius in radii {
        print!("Radius {:3} (100×100) ", radius);
        let time = benchmark_activation(grid_size, grid_size, radius);
        let affected_cells = ((radius * 2 + 1) * (radius * 2 + 1)) as f64 * std::f64::consts::PI / 4.0;
        println!("{:8.6}s  (~{:.0} cells affected)", time, affected_cells);
    }
    
    println!();
    println!("💡 Tips:");
    println!("  • Parallel processing activates for grids ≥ 50×50");
    println!("  • Parallel radius activates for radius ≥ 20");
    println!("  • Build with --release for accurate benchmarks");
    println!("  • Disable parallel: --no-default-features");
    println!();
    
    // Performance estimation
    println!("🚀 Performance Analysis");
    println!("═══════════════════════════════════════════════════════════");
    
    #[cfg(feature = "parallel")]
    {
        println!("With parallel processing enabled:");
        println!("  • Small grids (<50×50): Uses sequential (faster for small data)");
        println!("  • Large grids (≥50×50): Uses parallel (2-8x speedup)");
        println!("  • Speedup scales with available CPU cores");
    }
    
    #[cfg(not(feature = "parallel"))]
    {
        println!("Parallel processing is disabled.");
        println!("  • Enable with: cargo run --example benchmark --release");
        println!("  • Expected speedup: 2-8x on multi-core systems");
    }
    
    println!();
}
