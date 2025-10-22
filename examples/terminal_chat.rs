/// Terminal Chat Application with Persistent Toroidal Memory
/// 
/// A simple chatbot that uses toroidal memory to build conversation context.
/// Features:
/// - Persistent memory (saved to disk between sessions)
/// - Topic-based spatial organization
/// - Emotion tracking (simple sentiment analysis)
/// - Context-aware responses
/// - Memory decay over time
/// 
/// Run with: cargo run --example terminal_chat

use ai_toroidal_memory::{ToroidalMemory, DiffusionEngine, DiffusionConfig};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

// ============================================================================
// Memory State
// ============================================================================

#[derive(Serialize, Deserialize)]
struct ChatbotMemory {
    // Memory layers
    activation: Vec<Vec<f64>>,
    emotion: Vec<Vec<f64>>,
    confidence: Vec<Vec<f64>>,
    
    // Metadata
    position_labels: HashMap<String, (isize, isize)>, // topic -> position
    label_positions: HashMap<(isize, isize), String>, // position -> topic
    
    // User info
    user_id: String,
    turn_count: usize,
    
    // Config
    width: usize,
    height: usize,
}

impl ChatbotMemory {
    fn new(user_id: String, size: usize) -> Self {
        ChatbotMemory {
            activation: vec![vec![0.0; size]; size],
            emotion: vec![vec![0.0; size]; size],
            confidence: vec![vec![0.8; size]; size],
            position_labels: HashMap::new(),
            label_positions: HashMap::new(),
            user_id,
            turn_count: 0,
            width: size,
            height: size,
        }
    }
    
    fn save(&self, filepath: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(filepath, json)?;
        println!("💾 Memory saved to {}", filepath);
        Ok(())
    }
    
    fn load(filepath: &str) -> std::io::Result<Self> {
        let json = fs::read_to_string(filepath)?;
        let memory: ChatbotMemory = serde_json::from_str(&json)?;
        println!("📂 Memory loaded from {}", filepath);
        Ok(memory)
    }
}

// ============================================================================
// Chatbot with Toroidal Memory
// ============================================================================

struct ToroidalChatbot {
    activation_memory: ToroidalMemory<f64>,
    emotion_memory: ToroidalMemory<f64>,
    confidence_memory: ToroidalMemory<f64>,
    
    diffusion_engine: DiffusionEngine,
    
    position_labels: HashMap<String, (isize, isize)>,
    label_positions: HashMap<(isize, isize), String>,
    
    user_id: String,
    turn_count: usize,
    memory_file: String,
}

impl ToroidalChatbot {
    fn new(user_id: &str, size: usize) -> Self {
        let memory_file = format!("chat_memory_{}.json", user_id);
        
        // Try to load existing memory
        if Path::new(&memory_file).exists() {
            Self::load_from_file(&memory_file, user_id)
        } else {
            println!("🆕 Creating new memory for user: {}", user_id);
            ToroidalChatbot {
                activation_memory: ToroidalMemory::new(size, size),
                emotion_memory: ToroidalMemory::new(size, size),
                confidence_memory: ToroidalMemory::new(size, size),
                diffusion_engine: DiffusionEngine::new(DiffusionConfig {
                    diffusion_rate: 0.25,
                    decay_rate: 0.15,
                    threshold: 0.01,
                }),
                position_labels: HashMap::new(),
                label_positions: HashMap::new(),
                user_id: user_id.to_string(),
                turn_count: 0,
                memory_file,
            }
        }
    }
    
    fn load_from_file(filepath: &str, user_id: &str) -> Self {
        match ChatbotMemory::load(filepath) {
            Ok(saved) => {
                let mut bot = ToroidalChatbot {
                    activation_memory: ToroidalMemory::new(saved.width, saved.height),
                    emotion_memory: ToroidalMemory::new(saved.width, saved.height),
                    confidence_memory: ToroidalMemory::new(saved.width, saved.height),
                    diffusion_engine: DiffusionEngine::new(DiffusionConfig {
                        diffusion_rate: 0.25,
                        decay_rate: 0.15,
                        threshold: 0.01,
                    }),
                    position_labels: saved.position_labels,
                    label_positions: saved.label_positions,
                    user_id: saved.user_id,
                    turn_count: saved.turn_count,
                    memory_file: filepath.to_string(),
                };
                
                // Restore memory state
                for y in 0..saved.height {
                    for x in 0..saved.width {
                        bot.activation_memory.set(x as isize, y as isize, saved.activation[y][x]);
                        bot.emotion_memory.set(x as isize, y as isize, saved.emotion[y][x]);
                        bot.confidence_memory.set(x as isize, y as isize, saved.confidence[y][x]);
                    }
                }
                
                bot
            }
            Err(_) => {
                println!("⚠️  Could not load memory, creating new one");
                Self::new(user_id, 40)
            }
        }
    }
    
    fn save_to_file(&self) -> std::io::Result<()> {
        let (width, height) = self.activation_memory.dimensions();
        
        let mut activation = vec![vec![0.0; width]; height];
        let mut emotion = vec![vec![0.0; width]; height];
        let mut confidence = vec![vec![0.8; width]; height];
        
        for y in 0..height {
            for x in 0..width {
                activation[y][x] = *self.activation_memory.get(x as isize, y as isize).unwrap();
                emotion[y][x] = *self.emotion_memory.get(x as isize, y as isize).unwrap();
                confidence[y][x] = *self.confidence_memory.get(x as isize, y as isize).unwrap();
            }
        }
        
        let saved = ChatbotMemory {
            activation,
            emotion,
            confidence,
            position_labels: self.position_labels.clone(),
            label_positions: self.label_positions.clone(),
            user_id: self.user_id.clone(),
            turn_count: self.turn_count,
            width,
            height,
        };
        
        saved.save(&self.memory_file)
    }
    
    /// Hash a topic to a position in toroidal space
    fn topic_to_position(&self, topic: &str) -> (isize, isize) {
        let mut hasher = DefaultHasher::new();
        topic.hash(&mut hasher);
        let hash = hasher.finish();
        
        let (width, height) = self.activation_memory.dimensions();
        let x = (hash % width as u64) as isize;
        let y = ((hash / width as u64) % height as u64) as isize;
        
        (x, y)
    }
    
    /// Simple sentiment analysis (keyword-based)
    fn analyze_sentiment(&self, text: &str) -> f64 {
        let text_lower = text.to_lowercase();
        
        let positive_words = [
            "love", "great", "awesome", "wonderful", "excellent", "happy",
            "excited", "amazing", "fantastic", "good", "nice", "enjoy",
            "like", "thanks", "appreciate", "perfect", "beautiful"
        ];
        
        let negative_words = [
            "hate", "terrible", "awful", "bad", "horrible", "sad", "angry",
            "frustrated", "annoyed", "worried", "anxious", "stressed",
            "difficult", "problem", "issue", "hard", "confusing"
        ];
        
        let pos_count = positive_words.iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64;
        
        let neg_count = negative_words.iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64;
        
        if pos_count + neg_count == 0.0 {
            0.0 // Neutral
        } else {
            (pos_count - neg_count) / (pos_count + neg_count)
        }
    }
    
    /// Extract topics from message (simple word extraction)
    fn extract_topics(&self, text: &str) -> Vec<String> {
        let stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at",
                         "to", "for", "of", "with", "is", "was", "are", "be",
                         "i", "you", "me", "my", "your", "we", "us", "they"];
        
        text.to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3 && !stop_words.contains(w))
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .take(5)  // Limit to 5 topics per message
            .map(String::from)
            .collect()
    }
    
    /// Process a user message and update memory
    fn process_message(&mut self, message: &str) {
        self.turn_count += 1;
        
        // Extract topics and sentiment
        let topics = self.extract_topics(message);
        let sentiment = self.analyze_sentiment(message);
        
        // Store each topic in memory
        for topic in &topics {
            let pos = if let Some(&existing_pos) = self.position_labels.get(topic) {
                existing_pos
            } else {
                let new_pos = self.topic_to_position(topic);
                self.position_labels.insert(topic.clone(), new_pos);
                self.label_positions.insert(new_pos, topic.clone());
                new_pos
            };
            
            // Activate position with radius
            self.activate_region(pos.0, pos.1, 2, 1.0);
            
            // Store emotion
            self.emotion_memory.set(pos.0, pos.1, sentiment);
            
            // Update confidence (more mentions = higher confidence)
            let current_conf = *self.confidence_memory.get(pos.0, pos.1).unwrap();
            self.confidence_memory.set(pos.0, pos.1, (current_conf + 0.1).min(1.0));
        }
        
        // Run diffusion to spread activation
        self.diffusion_engine.step(&mut self.activation_memory);
    }
    
    /// Activate a region with radius
    fn activate_region(&mut self, x: isize, y: isize, radius: isize, strength: f64) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let dist = ((dx * dx + dy * dy) as f64).sqrt();
                if dist <= radius as f64 {
                    let activation_strength = strength * (1.0 - dist / (radius as f64 + 1.0));
                    let current = *self.activation_memory.get(x + dx, y + dy).unwrap();
                    self.activation_memory.set(
                        x + dx,
                        y + dy,
                        current.max(activation_strength)
                    );
                }
            }
        }
    }
    
    /// Get active topics from memory
    fn get_active_topics(&self, threshold: f64, limit: usize) -> Vec<(String, f64, f64)> {
        let mut topics = Vec::new();
        
        for (topic, &(x, y)) in &self.position_labels {
            let activation = *self.activation_memory.get(x, y).unwrap();
            if activation > threshold {
                let emotion = *self.emotion_memory.get(x, y).unwrap();
                topics.push((topic.clone(), activation, emotion));
            }
        }
        
        // Sort by activation strength
        topics.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        topics.truncate(limit);
        
        topics
    }
    
    /// Generate a context summary
    fn generate_context(&self) -> String {
        let active_topics = self.get_active_topics(0.3, 10);
        
        if active_topics.is_empty() {
            return "No active topics in memory yet.".to_string();
        }
        
        let mut context = String::new();
        context.push_str("📊 Current Memory State:\n\n");
        
        // Categorize by emotion
        let positive: Vec<_> = active_topics.iter()
            .filter(|(_, _, e)| *e > 0.3)
            .collect();
        
        let negative: Vec<_> = active_topics.iter()
            .filter(|(_, _, e)| *e < -0.3)
            .collect();
        
        let neutral: Vec<_> = active_topics.iter()
            .filter(|(_, _, e)| *e >= -0.3 && *e <= 0.3)
            .collect();
        
        if !positive.is_empty() {
            context.push_str("😊 Positive Topics:\n");
            for (topic, act, emo) in positive {
                context.push_str(&format!("  • {} (strength: {:.2}, sentiment: {:.2})\n", 
                    topic, act, emo));
            }
            context.push('\n');
        }
        
        if !negative.is_empty() {
            context.push_str("😟 Concerns/Challenges:\n");
            for (topic, act, emo) in negative {
                context.push_str(&format!("  • {} (strength: {:.2}, concern: {:.2})\n", 
                    topic, act, emo.abs()));
            }
            context.push('\n');
        }
        
        if !neutral.is_empty() {
            context.push_str("📝 Neutral Topics:\n");
            for (topic, act, _) in neutral {
                context.push_str(&format!("  • {} (strength: {:.2})\n", topic, act));
            }
        }
        
        context
    }
    
    /// Generate a simple response based on context
    fn generate_response(&self, message: &str) -> String {
        let active_topics = self.get_active_topics(0.3, 5);
        let sentiment = self.analyze_sentiment(message);
        
        // Check if message is a question
        let is_question = message.trim().ends_with('?');
        
        // Build response
        let mut response = String::new();
        
        if is_question {
            if message.to_lowercase().contains("what") && 
               (message.to_lowercase().contains("remember") || 
                message.to_lowercase().contains("know")) {
                // User asking what we remember
                if active_topics.is_empty() {
                    response.push_str("We haven't talked about much yet. What would you like to discuss?");
                } else {
                    response.push_str("I remember we've discussed: ");
                    let topics: Vec<String> = active_topics.iter()
                        .map(|(t, _, _)| t.clone())
                        .collect();
                    response.push_str(&topics.join(", "));
                    response.push('.');
                }
            } else {
                // General question
                if !active_topics.is_empty() {
                    let main_topic = &active_topics[0].0;
                    response.push_str(&format!(
                        "That's a good question! It seems related to {} which we've been discussing. ",
                        main_topic
                    ));
                }
                response.push_str("I'm a simple chatbot, but I'm happy to continue our conversation!");
            }
        } else {
            // Statement
            if sentiment > 0.5 {
                response.push_str("I'm glad to hear that! ");
            } else if sentiment < -0.5 {
                response.push_str("I understand that can be challenging. ");
            }
            
            if !active_topics.is_empty() {
                let topics: Vec<String> = active_topics.iter()
                    .take(2)
                    .map(|(t, _, _)| t.clone())
                    .collect();
                
                response.push_str(&format!(
                    "We've been talking about {}. What else would you like to explore?",
                    topics.join(" and ")
                ));
            } else {
                response.push_str("Tell me more!");
            }
        }
        
        response
    }
}

// ============================================================================
// Terminal UI
// ============================================================================

fn print_header() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║        Toroidal Memory Chatbot - Terminal Edition         ║");
    println!("║                                                            ║");
    println!("║  Your conversations are stored in spatial memory that     ║");
    println!("║  persists between sessions. Related topics cluster        ║");
    println!("║  together, and memories naturally fade over time.         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

fn print_help() {
    println!("\n📖 Commands:");
    println!("  /help      - Show this help message");
    println!("  /context   - Show current memory state");
    println!("  /stats     - Show memory statistics");
    println!("  /clear     - Clear all memory (cannot be undone!)");
    println!("  /save      - Manually save memory to disk");
    println!("  /quit      - Exit (automatically saves)\n");
}

fn print_stats(bot: &ToroidalChatbot) {
    println!("\n📈 Memory Statistics:");
    println!("  User ID: {}", bot.user_id);
    println!("  Conversation turns: {}", bot.turn_count);
    println!("  Topics stored: {}", bot.position_labels.len());
    
    let (width, height) = bot.activation_memory.dimensions();
    println!("  Memory size: {}×{} = {} cells", width, height, width * height);
    
    let active_count = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            *bot.activation_memory.get(x as isize, y as isize).unwrap() > 0.1
        })
        .count();
    
    println!("  Active cells: {} ({:.1}% of memory)", 
        active_count, 
        100.0 * active_count as f64 / (width * height) as f64
    );
    println!();
}

fn main() {
    print_header();
    
    let user_id = "user_001";  // Single user system
    let mut bot = ToroidalChatbot::new(user_id, 40);
    
    println!("👋 Hello! I'm your toroidal memory chatbot.");
    println!("💬 I'll remember our conversations and build context over time.");
    print_help();
    
    loop {
        // Prompt
        print!("\n💭 You: ");
        io::stdout().flush().unwrap();
        
        // Read input
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        // Handle commands
        match input {
            "/help" => {
                print_help();
                continue;
            }
            "/context" => {
                println!("\n{}", bot.generate_context());
                continue;
            }
            "/stats" => {
                print_stats(&bot);
                continue;
            }
            "/clear" => {
                print!("\n⚠️  Are you sure you want to clear all memory? (yes/no): ");
                io::stdout().flush().unwrap();
                let mut confirm = String::new();
                io::stdin().read_line(&mut confirm).unwrap();
                if confirm.trim().to_lowercase() == "yes" {
                    bot = ToroidalChatbot::new(user_id, 40);
                    // Delete the file
                    let _ = fs::remove_file(&bot.memory_file);
                    println!("✅ Memory cleared!");
                } else {
                    println!("❌ Cancelled.");
                }
                continue;
            }
            "/save" => {
                match bot.save_to_file() {
                    Ok(_) => println!("✅ Memory saved successfully!"),
                    Err(e) => println!("❌ Error saving: {}", e),
                }
                continue;
            }
            "/quit" | "/exit" => {
                println!("\n👋 Saving memory and exiting...");
                if let Err(e) = bot.save_to_file() {
                    println!("⚠️  Warning: Could not save memory: {}", e);
                }
                println!("Goodbye! 👋\n");
                break;
            }
            _ => {}
        }
        
        // Process message
        bot.process_message(input);
        
        // Generate response
        let response = bot.generate_response(input);
        println!("🤖 Bot: {}", response);
        
        // Auto-save every 5 turns
        if bot.turn_count % 5 == 0 {
            let _ = bot.save_to_file();
        }
    }
}
