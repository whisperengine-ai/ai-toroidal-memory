mod toroidal_memory;
mod pattern_matcher;
mod diffusion;

use toroidal_memory::ToroidalMemory;
use pattern_matcher::{Pattern, PatternMatcher};
use diffusion::DiffusionEngine;

fn main() {
    println!("=== AI Toroidal Memory Explorer ===\n");

    // Example 1: Basic toroidal memory
    demo_basic_memory();
    
    // Example 2: Pattern matching
    demo_pattern_matching();
    
    // Example 3: Diffusion
    demo_diffusion();
}

fn demo_basic_memory() {
    println!("--- Demo 1: Basic Toroidal Memory ---");
    
    let mut memory: ToroidalMemory<i32> = ToroidalMemory::new(5, 5);
    
    // Set some values
    memory.set(0, 0, 1);
    memory.set(2, 2, 5);
    memory.set(4, 4, 9);
    
    println!("Memory grid:");
    println!("{}", memory);
    
    // Demonstrate wrapping
    println!("Value at (0, 0): {}", memory.get(0, 0).unwrap());
    println!("Value at (-5, -5) (wrapped): {}", memory.get(-5, -5).unwrap());
    println!("Value at (5, 5) (wrapped): {}", memory.get(5, 5).unwrap());
    
    // Get neighbors
    let neighbors = memory.get_neighbors(2, 2, 1);
    println!("Number of neighbors around (2, 2): {}\n", neighbors.len());
}

fn demo_pattern_matching() {
    println!("--- Demo 2: Pattern Matching ---");
    
    let mut memory = ToroidalMemory::new(8, 8);
    
    // Create a simple pattern (L-shape)
    memory.set(2, 2, 1);
    memory.set(2, 3, 1);
    memory.set(2, 4, 1);
    memory.set(3, 4, 1);
    
    // Duplicate pattern elsewhere
    memory.set(5, 5, 1);
    memory.set(5, 6, 1);
    memory.set(5, 7, 1);
    memory.set(6, 7, 1);
    
    println!("Memory with L-patterns:");
    println!("{}", memory);
    
    // Define the L-pattern
    let pattern = Pattern::new(2, 3, vec![
        1, 0,
        1, 0,
        1, 1,
    ]);
    
    // Find all matches
    let matches = PatternMatcher::find_pattern(&memory, &pattern);
    println!("Found {} matches of the L-pattern", matches.len());
    for (x, y) in &matches {
        println!("  Match at position ({}, {})", x, y);
    }
    println!();
}

fn demo_diffusion() {
    println!("--- Demo 3: Diffusion Simulation ---");
    
    let mut memory = ToroidalMemory::new(10, 10);
    
    // Add initial activations
    DiffusionEngine::activate_radius(&mut memory, 5, 5, 2, 1.0);
    
    println!("Initial activation:");
    print_activation_grid(&memory);
    
    // Run diffusion
    let engine = DiffusionEngine::with_defaults();
    engine.run(&mut memory, 5);
    
    println!("\nAfter 5 diffusion steps:");
    print_activation_grid(&memory);
    
    println!("\nDiffusion spreads activation across the toroidal surface!");
    println!("Notice how activation wraps around edges.\n");
}

fn print_activation_grid(memory: &ToroidalMemory<f64>) {
    let (width, height) = memory.dimensions();
    for y in 0..height {
        for x in 0..width {
            let value = memory.get(x as isize, y as isize).unwrap();
            let symbol = if *value > 0.7 {
                "█"
            } else if *value > 0.4 {
                "▓"
            } else if *value > 0.2 {
                "▒"
            } else if *value > 0.05 {
                "░"
            } else {
                "·"
            };
            print!("{}", symbol);
        }
        println!();
    }
}
