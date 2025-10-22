/// Advanced examples for AI Toroidal Memory
/// 
/// Run with: cargo run --example advanced

use ai_toroidal_memory::{ToroidalMemory, DiffusionEngine};

fn main() {
    println!("=== Advanced Toroidal Memory Examples ===\n");
    
    example_1_wrapping_world();
    example_2_cellular_automaton();
    example_3_attention_mechanism();
}

/// Example 1: Simulating a wrapping world for game AI
fn example_1_wrapping_world() {
    println!("--- Example 1: Wrapping World Navigation ---");
    println!("Simulating an agent navigating a toroidal world\n");
    
    let mut world = ToroidalMemory::new(15, 10);
    
    // Place obstacles
    for x in 3..8 {
        world.set(x, 3, '#');
    }
    for y in 5..8 {
        world.set(10, y, '#');
    }
    
    // Agent starting position
    let mut agent_x = 0isize;
    let mut agent_y = 0isize;
    
    println!("Initial world:");
    print_world(&world, agent_x, agent_y);
    
    // Agent moves in a pattern that wraps around
    let moves = vec![(5, 0), (5, 5), (-3, 2), (10, -2)];
    
    for (dx, dy) in moves {
        agent_x += dx;
        agent_y += dy;
        println!("\nAgent moves by ({}, {}). New position: ({}, {})", dx, dy, agent_x, agent_y);
        println!("(Wrapped coordinates: {:?})", wrap_coord(agent_x, agent_y, 15, 10));
    }
    
    println!();
}

/// Example 2: Simple cellular automaton on toroidal grid
fn example_2_cellular_automaton() {
    println!("--- Example 2: Cellular Automaton (Toroidal Life) ---");
    println!("Conway's Game of Life on a torus\n");
    
    let mut grid = ToroidalMemory::new(20, 20);
    
    // Initialize with a glider pattern
    grid.set(5, 5, 1);
    grid.set(6, 6, 1);
    grid.set(4, 7, 1);
    grid.set(5, 7, 1);
    grid.set(6, 7, 1);
    
    println!("Generation 0:");
    print_life_grid(&grid);
    
    // Simulate 5 generations
    for generation in 1..=5 {
        grid = life_step(&grid);
        println!("\nGeneration {}:", generation);
        print_life_grid(&grid);
    }
    
    println!("\nNotice: The glider wraps around edges seamlessly!\n");
}

/// Example 3: Attention-like spreading activation
fn example_3_attention_mechanism() {
    println!("--- Example 3: Attention Mechanism Simulation ---");
    println!("Simulating attention spreading across memory\n");
    
    let mut memory = ToroidalMemory::new(25, 25);
    
    // Create multiple points of interest
    DiffusionEngine::activate_radius(&mut memory, 8, 8, 2, 1.0);
    DiffusionEngine::activate_radius(&mut memory, 18, 18, 2, 0.8);
    DiffusionEngine::activate_radius(&mut memory, 3, 20, 2, 0.6);
    
    println!("Initial activations (3 regions):");
    print_attention_grid(&memory);
    
    // Spread attention
    let engine = DiffusionEngine::with_defaults();
    engine.run(&mut memory, 10);
    
    println!("\nAfter attention spreading:");
    print_attention_grid(&memory);
    
    println!("\nActivation fields overlap and wrap around the toroidal surface.");
    println!("This could model how attention spreads in spatial memory!\n");
}

// Helper functions

fn wrap_coord(x: isize, y: isize, width: usize, height: usize) -> (usize, usize) {
    let wx = ((x % width as isize + width as isize) % width as isize) as usize;
    let wy = ((y % height as isize + height as isize) % height as isize) as usize;
    (wx, wy)
}

fn print_world(world: &ToroidalMemory<char>, agent_x: isize, agent_y: isize) {
    let (width, height) = world.dimensions();
    let (ax, ay) = wrap_coord(agent_x, agent_y, width, height);
    
    for y in 0..height {
        for x in 0..width {
            if x == ax && y == ay {
                print!("@");
            } else {
                let cell = world.get(x as isize, y as isize).unwrap();
                print!("{}", if *cell == '\0' { '.' } else { *cell });
            }
        }
        println!();
    }
}

fn life_step(grid: &ToroidalMemory<u8>) -> ToroidalMemory<u8> {
    let (width, height) = grid.dimensions();
    let mut new_grid = ToroidalMemory::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let alive = *grid.get(x as isize, y as isize).unwrap() == 1;
            let neighbors = count_life_neighbors(grid, x as isize, y as isize);
            
            let new_state = match (alive, neighbors) {
                (true, 2) | (true, 3) => 1,
                (false, 3) => 1,
                _ => 0,
            };
            
            new_grid.set(x as isize, y as isize, new_state);
        }
    }
    
    new_grid
}

fn count_life_neighbors(grid: &ToroidalMemory<u8>, x: isize, y: isize) -> usize {
    let offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];
    
    offsets.iter()
        .filter(|(dx, dy)| {
            *grid.get(x + dx, y + dy).unwrap() == 1
        })
        .count()
}

fn print_life_grid(grid: &ToroidalMemory<u8>) {
    let (width, height) = grid.dimensions();
    for y in 0..height.min(15) {
        for x in 0..width.min(30) {
            let cell = grid.get(x as isize, y as isize).unwrap();
            print!("{}", if *cell == 1 { "█" } else { "·" });
        }
        println!();
    }
}

fn print_attention_grid(memory: &ToroidalMemory<f64>) {
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
