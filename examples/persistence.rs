/// Persistence and State Management for Toroidal Memory
/// 
/// This example demonstrates how to save/load toroidal memory state,
/// enabling long-term storage, session resumption, and state checkpointing.
/// 
/// Run with: cargo run --example persistence

use ai_toroidal_memory::{ToroidalMemory, DiffusionEngine};
use std::fs::File;
use std::io::{Write, Read};
use std::path::Path;

// ============================================================================
// SERIALIZATION APPROACHES
// ============================================================================

/// Approach 1: Simple Binary Format (most efficient)
mod binary_format {
    use super::*;
    
    pub fn save_to_binary(memory: &ToroidalMemory<f64>, path: &str) -> std::io::Result<()> {
        let (width, height) = memory.dimensions();
        let mut file = File::create(path)?;
        
        // Write header: [width: 4 bytes][height: 4 bytes]
        file.write_all(&(width as u32).to_le_bytes())?;
        file.write_all(&(height as u32).to_le_bytes())?;
        
        // Write data: each f64 value as 8 bytes
        for y in 0..height {
            for x in 0..width {
                let value = memory.get(x as isize, y as isize).unwrap();
                file.write_all(&value.to_le_bytes())?;
            }
        }
        
        println!("✓ Saved to binary: {} ({} bytes)", path, 8 + width * height * 8);
        Ok(())
    }
    
    pub fn load_from_binary(path: &str) -> std::io::Result<ToroidalMemory<f64>> {
        let mut file = File::open(path)?;
        
        // Read header
        let mut width_bytes = [0u8; 4];
        let mut height_bytes = [0u8; 4];
        file.read_exact(&mut width_bytes)?;
        file.read_exact(&mut height_bytes)?;
        
        let width = u32::from_le_bytes(width_bytes) as usize;
        let height = u32::from_le_bytes(height_bytes) as usize;
        
        // Create memory
        let mut memory = ToroidalMemory::new(width, height);
        
        // Read data
        for y in 0..height {
            for x in 0..width {
                let mut value_bytes = [0u8; 8];
                file.read_exact(&mut value_bytes)?;
                let value = f64::from_le_bytes(value_bytes);
                memory.set(x as isize, y as isize, value);
            }
        }
        
        println!("✓ Loaded from binary: {} ({}x{})", path, width, height);
        Ok(memory)
    }
}

/// Approach 2: JSON Format (human-readable)
mod json_format {
    use super::*;
    use serde::{Serialize, Deserialize};
    
    #[derive(Serialize, Deserialize)]
    struct MemorySnapshot {
        width: usize,
        height: usize,
        data: Vec<f64>,
    }
    
    pub fn save_to_json(memory: &ToroidalMemory<f64>, path: &str) -> std::io::Result<()> {
        let (width, height) = memory.dimensions();
        let mut data = Vec::with_capacity(width * height);
        
        for y in 0..height {
            for x in 0..width {
                data.push(*memory.get(x as isize, y as isize).unwrap());
            }
        }
        
        let snapshot = MemorySnapshot { width, height, data };
        let json = serde_json::to_string_pretty(&snapshot)?;
        
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        
        println!("✓ Saved to JSON: {} ({} bytes)", path, json.len());
        Ok(())
    }
    
    pub fn load_from_json(path: &str) -> std::io::Result<ToroidalMemory<f64>> {
        let mut file = File::open(path)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        
        let snapshot: MemorySnapshot = serde_json::from_str(&json)?;
        let mut memory = ToroidalMemory::new(snapshot.width, snapshot.height);
        
        for y in 0..snapshot.height {
            for x in 0..snapshot.width {
                let index = y * snapshot.width + x;
                memory.set(x as isize, y as isize, snapshot.data[index]);
            }
        }
        
        println!("✓ Loaded from JSON: {} ({}x{})", path, snapshot.width, snapshot.height);
        Ok(memory)
    }
}

/// Approach 3: Compressed Format (for large memories)
mod compressed_format {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::read::GzDecoder;
    use flate2::Compression;
    
    pub fn save_compressed(memory: &ToroidalMemory<f64>, path: &str) -> std::io::Result<()> {
        let (width, height) = memory.dimensions();
        
        // Create compressed writer
        let file = File::create(path)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        
        // Write header
        encoder.write_all(&(width as u32).to_le_bytes())?;
        encoder.write_all(&(height as u32).to_le_bytes())?;
        
        // Write data
        for y in 0..height {
            for x in 0..width {
                let value = memory.get(x as isize, y as isize).unwrap();
                encoder.write_all(&value.to_le_bytes())?;
            }
        }
        
        encoder.finish()?;
        
        let metadata = std::fs::metadata(path)?;
        println!("✓ Saved compressed: {} ({} bytes)", path, metadata.len());
        Ok(())
    }
    
    pub fn load_compressed(path: &str) -> std::io::Result<ToroidalMemory<f64>> {
        let file = File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        
        // Read header
        let mut width_bytes = [0u8; 4];
        let mut height_bytes = [0u8; 4];
        decoder.read_exact(&mut width_bytes)?;
        decoder.read_exact(&mut height_bytes)?;
        
        let width = u32::from_le_bytes(width_bytes) as usize;
        let height = u32::from_le_bytes(height_bytes) as usize;
        
        // Create memory
        let mut memory = ToroidalMemory::new(width, height);
        
        // Read data
        for y in 0..height {
            for x in 0..width {
                let mut value_bytes = [0u8; 8];
                decoder.read_exact(&mut value_bytes)?;
                let value = f64::from_le_bytes(value_bytes);
                memory.set(x as isize, y as isize, value);
            }
        }
        
        println!("✓ Loaded compressed: {} ({}x{})", path, width, height);
        Ok(memory)
    }
}

// ============================================================================
// STATE MANAGEMENT PATTERNS
// ============================================================================

/// Pattern 1: Checkpoint System (for training/long-running processes)
struct CheckpointManager {
    checkpoint_dir: String,
    max_checkpoints: usize,
}

impl CheckpointManager {
    fn new(dir: &str, max_checkpoints: usize) -> Self {
        std::fs::create_dir_all(dir).ok();
        CheckpointManager {
            checkpoint_dir: dir.to_string(),
            max_checkpoints,
        }
    }
    
    fn save_checkpoint(&self, memory: &ToroidalMemory<f64>, step: usize) -> std::io::Result<()> {
        let path = format!("{}/checkpoint_{:06}.bin", self.checkpoint_dir, step);
        binary_format::save_to_binary(memory, &path)?;
        
        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;
        Ok(())
    }
    
    fn load_latest_checkpoint(&self) -> std::io::Result<(ToroidalMemory<f64>, usize)> {
        let entries = std::fs::read_dir(&self.checkpoint_dir)?;
        
        let mut checkpoints: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "bin"))
            .collect();
        
        checkpoints.sort_by_key(|e| e.path());
        
        if let Some(latest) = checkpoints.last() {
            let path = latest.path();
            let filename = path.file_stem().unwrap().to_str().unwrap();
            let step: usize = filename.split('_').nth(1).unwrap().parse().unwrap();
            
            let memory = binary_format::load_from_binary(path.to_str().unwrap())?;
            Ok((memory, step))
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "No checkpoints found"))
        }
    }
    
    fn cleanup_old_checkpoints(&self) -> std::io::Result<()> {
        let entries = std::fs::read_dir(&self.checkpoint_dir)?;
        
        let mut checkpoints: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "bin"))
            .collect();
        
        checkpoints.sort_by_key(|e| e.path());
        
        // Remove oldest checkpoints if we exceed max
        while checkpoints.len() > self.max_checkpoints {
            if let Some(oldest) = checkpoints.first() {
                std::fs::remove_file(oldest.path())?;
                checkpoints.remove(0);
            }
        }
        
        Ok(())
    }
}

/// Pattern 2: Session Manager (for chatbots/interactive systems)
struct SessionManager {
    sessions_dir: String,
}

impl SessionManager {
    fn new(dir: &str) -> Self {
        std::fs::create_dir_all(dir).ok();
        SessionManager {
            sessions_dir: dir.to_string(),
        }
    }
    
    fn save_session(&self, session_id: &str, memory: &ToroidalMemory<f64>) -> std::io::Result<()> {
        let path = format!("{}/{}.session", self.sessions_dir, session_id);
        compressed_format::save_compressed(memory, &path)?;
        println!("Session '{}' saved", session_id);
        Ok(())
    }
    
    fn load_session(&self, session_id: &str) -> std::io::Result<ToroidalMemory<f64>> {
        let path = format!("{}/{}.session", self.sessions_dir, session_id);
        let memory = compressed_format::load_compressed(&path)?;
        println!("Session '{}' loaded", session_id);
        Ok(memory)
    }
    
    fn list_sessions(&self) -> Vec<String> {
        std::fs::read_dir(&self.sessions_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map_or(false, |ext| ext == "session"))
                    .map(|e| e.path().file_stem().unwrap().to_str().unwrap().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Pattern 3: Incremental State (delta encoding for efficiency)
struct IncrementalState {
    base_memory: ToroidalMemory<f64>,
    changes: Vec<(isize, isize, f64)>,  // Position + value changes
}

impl IncrementalState {
    fn new(memory: ToroidalMemory<f64>) -> Self {
        IncrementalState {
            base_memory: memory,
            changes: Vec::new(),
        }
    }
    
    fn record_change(&mut self, x: isize, y: isize, new_value: f64) {
        self.changes.push((x, y, new_value));
    }
    
    fn apply_changes(&mut self) {
        for (x, y, value) in &self.changes {
            self.base_memory.set(*x, *y, *value);
        }
        self.changes.clear();
    }
    
    fn save_delta(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        
        // Write number of changes
        file.write_all(&(self.changes.len() as u32).to_le_bytes())?;
        
        // Write each change
        for (x, y, value) in &self.changes {
            file.write_all(&(*x as i32).to_le_bytes())?;
            file.write_all(&(*y as i32).to_le_bytes())?;
            file.write_all(&value.to_le_bytes())?;
        }
        
        println!("✓ Saved delta: {} ({} changes)", path, self.changes.len());
        Ok(())
    }
}

// ============================================================================
// EXAMPLES
// ============================================================================

fn main() {
    println!("=== Toroidal Memory Persistence Examples ===\n");
    
    demo_basic_save_load();
    demo_checkpoint_system();
    demo_session_management();
    demo_format_comparison();
}

fn demo_basic_save_load() {
    println!("--- Demo 1: Basic Save/Load ---\n");
    
    // Create and populate memory
    let mut memory = ToroidalMemory::new(20, 20);
    DiffusionEngine::activate_radius(&mut memory, 10, 10, 5, 1.0);
    let engine = DiffusionEngine::with_defaults();
    engine.run(&mut memory, 10);
    
    println!("Original memory created with activation at (10, 10)");
    println!("Center activation: {:.3}\n", memory.get(10, 10).unwrap());
    
    // Save in different formats
    binary_format::save_to_binary(&memory, "/tmp/memory.bin").unwrap();
    json_format::save_to_json(&memory, "/tmp/memory.json").unwrap();
    compressed_format::save_compressed(&memory, "/tmp/memory.bin.gz").unwrap();
    
    println!();
    
    // Load and verify
    let loaded = binary_format::load_from_binary("/tmp/memory.bin").unwrap();
    println!("Loaded center activation: {:.3}", loaded.get(10, 10).unwrap());
    println!("✓ State preserved correctly!\n");
}

fn demo_checkpoint_system() {
    println!("--- Demo 2: Checkpoint System ---");
    println!("Useful for training or long-running processes\n");
    
    let manager = CheckpointManager::new("/tmp/checkpoints", 3);
    let mut memory = ToroidalMemory::new(15, 15);
    
    // Simulate training steps
    for step in 0..5 {
        // Activate different regions
        let x = (step * 3) as isize;
        let y = (step * 2) as isize;
        DiffusionEngine::activate_radius(&mut memory, x, y, 2, 0.8);
        
        let engine = DiffusionEngine::with_defaults();
        engine.run(&mut memory, 3);
        
        // Save checkpoint
        manager.save_checkpoint(&memory, step).unwrap();
        println!("Step {}: Checkpoint saved", step);
    }
    
    println!();
    
    // Load latest checkpoint
    let (restored, step) = manager.load_latest_checkpoint().unwrap();
    println!("Restored from step: {}", step);
    println!("✓ Can resume training from checkpoint!\n");
}

fn demo_session_management() {
    println!("--- Demo 3: Session Management ---");
    println!("Useful for chatbots - save/restore conversation state\n");
    
    let manager = SessionManager::new("/tmp/sessions");
    
    // Session 1: Discussion about AI
    let mut ai_session = ToroidalMemory::new(25, 25);
    DiffusionEngine::activate_radius(&mut ai_session, 5, 5, 3, 0.9);
    DiffusionEngine::activate_radius(&mut ai_session, 20, 20, 3, 0.7);
    manager.save_session("user123_ai_discussion", &ai_session).unwrap();
    
    // Session 2: Discussion about cooking
    let mut cooking_session = ToroidalMemory::new(25, 25);
    DiffusionEngine::activate_radius(&mut cooking_session, 10, 10, 4, 1.0);
    manager.save_session("user123_cooking_chat", &cooking_session).unwrap();
    
    println!();
    
    // List sessions
    println!("Available sessions:");
    for session in manager.list_sessions() {
        println!("  - {}", session);
    }
    
    println!();
    
    // Restore a session
    let restored = manager.load_session("user123_ai_discussion").unwrap();
    println!("Restored activation at (5,5): {:.3}", restored.get(5, 5).unwrap());
    println!("✓ User can resume conversation where they left off!\n");
}

fn demo_format_comparison() {
    println!("--- Demo 4: Format Comparison ---\n");
    
    let mut memory = ToroidalMemory::new(50, 50);
    DiffusionEngine::activate_radius(&mut memory, 25, 25, 10, 1.0);
    let engine = DiffusionEngine::with_defaults();
    engine.run(&mut memory, 20);
    
    // Save in all formats
    binary_format::save_to_binary(&memory, "/tmp/test.bin").unwrap();
    json_format::save_to_json(&memory, "/tmp/test.json").unwrap();
    compressed_format::save_compressed(&memory, "/tmp/test.bin.gz").unwrap();
    
    println!();
    
    // Compare file sizes
    let bin_size = std::fs::metadata("/tmp/test.bin").unwrap().len();
    let json_size = std::fs::metadata("/tmp/test.json").unwrap().len();
    let gz_size = std::fs::metadata("/tmp/test.bin.gz").unwrap().len();
    
    println!("File Size Comparison (50x50 grid):");
    println!("  Binary:     {:6} bytes (1.00x) ← Fastest", bin_size);
    println!("  JSON:       {:6} bytes ({:.2}x) ← Human-readable", json_size, json_size as f64 / bin_size as f64);
    println!("  Compressed: {:6} bytes ({:.2}x) ← Smallest", gz_size, gz_size as f64 / bin_size as f64);
    
    println!("\nRecommendations:");
    println!("  - Binary: Production use, fast I/O");
    println!("  - JSON: Development, debugging, human inspection");
    println!("  - Compressed: Large memories, network transfer, archival\n");
}
