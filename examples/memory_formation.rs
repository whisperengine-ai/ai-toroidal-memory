/// Example demonstrating how memories are formed and retrieved
/// 
/// Run with: cargo run --example memory_formation

use ai_toroidal_memory::{ToroidalMemory, DiffusionEngine, DiffusionConfig};
use std::collections::HashMap;

fn main() {
    println!("=== Memory Formation in Toroidal Space ===\n");
    
    demo_episodic_memory();
    demo_associative_learning();
    demo_interference_and_consolidation();
}

/// Simulate storing episodic memories (events at different locations)
fn demo_episodic_memory() {
    println!("--- Demo 1: Episodic Memory Formation ---");
    println!("Simulating how an agent stores experiences in space\n");
    
    let mut memory = ToroidalMemory::new(30, 30);
    let mut memory_log = HashMap::new();
    
    // Simulate experiences over time
    let episodes = vec![
        ("Found food", 5, 5),
        ("Saw predator", 20, 20),
        ("Safe shelter", 10, 15),
        ("Water source", 25, 8),
    ];
    
    println!("Storing episodic memories:");
    for (i, (event, x, y)) in episodes.iter().enumerate() {
        println!("  Episode {}: '{}' at ({}, {})", i + 1, event, x, y);
        
        // Store memory with spreading activation
        DiffusionEngine::activate_radius(&mut memory, *x, *y, 3, 1.0);
        memory_log.insert((*x, *y), event);
        
        // Let it consolidate
        let engine = DiffusionEngine::with_defaults();
        engine.run(&mut memory, 3);
    }
    
    println!("\nMemory landscape after storing episodes:");
    print_memory_landscape(&memory);
    
    // Retrieval: Query nearby location
    let query_x = 6;
    let query_y = 6;
    println!("\nQuerying location ({}, {}):", query_x, query_y);
    println!("Activation strength: {:.3}", memory.get(query_x, query_y).unwrap());
    println!("This suggests proximity to: 'Found food' at (5, 5)");
    println!("Spatial memory allows location-based retrieval!\n");
}

/// Demonstrate how related memories become associated through co-activation
fn demo_associative_learning() {
    println!("--- Demo 2: Associative Memory Formation ---");
    println!("Demonstrating Hebbian-like learning: patterns that co-occur link together\n");
    
    let mut memory = ToroidalMemory::new(40, 40);
    
    // Two initially separate concepts
    println!("Initial state: Two separate memories");
    println!("  Memory A: 'Apple' at (10, 10)");
    println!("  Memory B: 'Red' at (30, 30)");
    
    DiffusionEngine::activate_radius(&mut memory, 10, 10, 2, 0.8);
    DiffusionEngine::activate_radius(&mut memory, 30, 30, 2, 0.8);
    
    print_memory_landscape(&memory);
    
    // Co-activation strengthens path between them
    println!("\nLearning phase: Co-activating both memories (apple → red association)");
    for trial in 1..=5 {
        println!("  Trial {}: Activating both memories simultaneously", trial);
        
        // Activate both
        DiffusionEngine::activate_radius(&mut memory, 10, 10, 2, 0.6);
        DiffusionEngine::activate_radius(&mut memory, 30, 30, 2, 0.6);
        
        // Diffusion creates path between them
        let engine = DiffusionEngine::with_defaults();
        engine.run(&mut memory, 4);
    }
    
    println!("\nAfter associative learning:");
    print_memory_landscape(&memory);
    println!("\nNotice: A gradient path now connects the two memories!");
    println!("Activating 'Apple' will spread to 'Red' through association.\n");
}

/// Show how similar memories can interfere and require consolidation
fn demo_interference_and_consolidation() {
    println!("--- Demo 3: Memory Interference and Consolidation ---");
    println!("Similar memories stored nearby can interfere with each other\n");
    
    let mut memory = ToroidalMemory::new(25, 25);
    
    // Store similar memories close together (high interference)
    println!("Scenario 1: High interference - storing similar memories close together");
    DiffusionEngine::activate_radius(&mut memory, 12, 12, 3, 1.0);
    println!("  Stored: Memory A at (12, 12)");
    
    let mut engine = DiffusionEngine::new(DiffusionConfig {
        diffusion_rate: 0.25,
        decay_rate: 0.05,
        threshold: 0.01,
    });
    engine.run(&mut memory, 2);
    
    // Store very similar memory nearby
    DiffusionEngine::activate_radius(&mut memory, 14, 14, 3, 1.0);
    println!("  Stored: Memory B at (14, 14) - only 2 units away!");
    engine.run(&mut memory, 2);
    
    println!("\nMemory state with interference:");
    print_memory_landscape(&memory);
    println!("\nProblem: The two memories blend together - hard to retrieve distinctly!");
    
    // Solution: Better encoding or consolidation
    println!("\n--- Consolidation Strategy ---");
    let mut consolidated = ToroidalMemory::new(25, 25);
    
    // Store with more separation
    DiffusionEngine::activate_radius(&mut consolidated, 8, 8, 3, 1.0);
    println!("  Memory A at (8, 8)");
    engine.run(&mut consolidated, 5);
    
    DiffusionEngine::activate_radius(&mut consolidated, 18, 18, 3, 1.0);
    println!("  Memory B at (18, 18) - better separation!");
    engine.run(&mut consolidated, 5);
    
    println!("\nConsolidated memory (with proper separation):");
    print_memory_landscape(&consolidated);
    println!("\nNow memories are distinct and can be retrieved without confusion!");
    
    println!("\n=== Key Insights ===");
    println!("1. Spatial separation prevents interference");
    println!("2. Diffusion naturally clusters related memories");
    println!("3. Consolidation (via diffusion) sharpens memory traces");
    println!("4. This mirrors biological memory consolidation during sleep!\n");
}

fn print_memory_landscape(memory: &ToroidalMemory<f64>) {
    let (width, height) = memory.dimensions();
    for y in 0..height {
        for x in 0..width {
            let value = memory.get(x as isize, y as isize).unwrap();
            let symbol = if *value > 0.7 {
                "█"
            } else if *value > 0.5 {
                "▓"
            } else if *value > 0.3 {
                "▒"
            } else if *value > 0.15 {
                "░"
            } else if *value > 0.05 {
                "·"
            } else {
                " "
            };
            print!("{}", symbol);
        }
        println!();
    }
}

/// Example of a more sophisticated memory encoding strategy
#[allow(dead_code)]
fn semantic_hash_position(concept: &str, width: usize, height: usize) -> (isize, isize) {
    // Simple hash function to map concepts to spatial positions
    // In real system, this could use embeddings or learned positions
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    concept.hash(&mut hasher);
    let hash = hasher.finish();
    
    let x = (hash % width as u64) as isize;
    let y = ((hash / width as u64) % height as u64) as isize;
    
    (x, y)
}

/// Example: Store a complex memory with multiple features
#[allow(dead_code)]
fn store_complex_memory(
    memory: &mut ToroidalMemory<f64>,
    features: Vec<(&str, f64)>,  // (feature_name, importance)
    width: usize,
    height: usize,
) {
    for (feature, importance) in features {
        let (x, y) = semantic_hash_position(feature, width, height);
        DiffusionEngine::activate_radius(memory, x, y, 2, importance);
    }
    
    // Let features associate through diffusion
    let engine = DiffusionEngine::with_defaults();
    engine.run(memory, 5);
}
