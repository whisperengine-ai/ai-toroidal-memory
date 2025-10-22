/// Practical Chatbot Application using Toroidal Memory
/// 
/// This example demonstrates how a conversational AI could use toroidal
/// memory for context management, topic tracking, and conversational flow.
/// 
/// Run with: cargo run --example chatbot

use ai_toroidal_memory::{ToroidalMemory, DiffusionEngine, DiffusionConfig};
use std::collections::HashMap;

/// Represents a chatbot with toroidal memory for conversation context
struct ToroidalChatbot {
    // Main conversation memory - tracks topics and context
    conversation_memory: ToroidalMemory<f64>,
    
    // Maps topics to spatial positions
    topic_positions: HashMap<String, (isize, isize)>,
    
    // Current focus of attention
    current_focus: (isize, isize),
    
    // Conversation history for demo
    conversation_log: Vec<String>,
}

impl ToroidalChatbot {
    fn new(memory_size: usize) -> Self {
        ToroidalChatbot {
            conversation_memory: ToroidalMemory::new(memory_size, memory_size),
            topic_positions: HashMap::new(),
            current_focus: (memory_size as isize / 2, memory_size as isize / 2),
            conversation_log: Vec::new(),
        }
    }
    
    /// Hash a topic to a spatial position (deterministic)
    fn topic_to_position(&self, topic: &str) -> (isize, isize) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        topic.hash(&mut hasher);
        let hash = hasher.finish();
        
        let (width, height) = self.conversation_memory.dimensions();
        let x = (hash % width as u64) as isize;
        let y = ((hash / width as u64) % height as u64) as isize;
        
        (x, y)
    }
    
    /// Process a user message and activate relevant topics
    fn process_message(&mut self, message: &str, topics: Vec<&str>) {
        println!("User: {}", message);
        self.conversation_log.push(format!("User: {}", message));
        
        // Activate all mentioned topics
        for topic in &topics {
            let pos = if let Some(&pos) = self.topic_positions.get(*topic) {
                pos
            } else {
                let pos = self.topic_to_position(topic);
                self.topic_positions.insert(topic.to_string(), pos);
                pos
            };
            
            // Activate topic in memory
            DiffusionEngine::activate_radius(
                &mut self.conversation_memory,
                pos.0,
                pos.1,
                3,
                0.8
            );
            
            println!("  [Activated topic: '{}' at position ({}, {})]", topic, pos.0, pos.1);
        }
        
        // Run diffusion to let topics associate
        let engine = DiffusionEngine::new(DiffusionConfig {
            diffusion_rate: 0.3,
            decay_rate: 0.02,  // Slow decay = longer memory
            threshold: 0.01,
        });
        engine.run(&mut self.conversation_memory, 3);
    }
    
    /// Generate response based on active memory regions
    fn generate_response(&mut self) -> String {
        // Find most active regions (topics with highest activation)
        let active_topics = self.find_active_topics(3);
        
        if active_topics.is_empty() {
            return "I'm not sure what to say about that.".to_string();
        }
        
        // Generate response mentioning active topics
        let response = format!(
            "That's interesting! Based on our conversation, I'm thinking about {} topics right now.",
            active_topics.len()
        );
        
        println!("Bot: {}", response);
        println!("  [Active topics in memory: {:?}]", active_topics);
        self.conversation_log.push(format!("Bot: {}", response));
        
        response
    }
    
    /// Find topics with highest activation
    fn find_active_topics(&self, top_n: usize) -> Vec<String> {
        let mut topic_activations: Vec<(String, f64)> = self.topic_positions
            .iter()
            .map(|(topic, &(x, y))| {
                let activation = *self.conversation_memory.get(x, y).unwrap();
                (topic.clone(), activation)
            })
            .collect();
        
        topic_activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        topic_activations
            .iter()
            .take(top_n)
            .filter(|(_, act)| *act > 0.1)
            .map(|(topic, _)| topic.clone())
            .collect()
    }
    
    /// Check if a topic is related to currently active topics (via proximity)
    fn is_topic_related(&self, topic: &str, threshold: f64) -> bool {
        if let Some(&(x, y)) = self.topic_positions.get(topic) {
            let activation = *self.conversation_memory.get(x, y).unwrap();
            activation > threshold
        } else {
            false
        }
    }
    
    /// Show memory state (for visualization)
    fn show_memory_state(&self, title: &str) {
        println!("\n{}", title);
        println!("{}", "=".repeat(title.len()));
        
        let (width, height) = self.conversation_memory.dimensions();
        for y in 0..height {
            for x in 0..width {
                let value = self.conversation_memory.get(x as isize, y as isize).unwrap();
                let symbol = if *value > 0.6 {
                    "█"
                } else if *value > 0.4 {
                    "▓"
                } else if *value > 0.2 {
                    "▒"
                } else if *value > 0.1 {
                    "░"
                } else {
                    "·"
                };
                print!("{}", symbol);
            }
            println!();
        }
        println!();
    }
}

fn main() {
    println!("=== Toroidal Memory Chatbot Demo ===\n");
    
    demo_context_management();
    demo_topic_association();
    demo_conversation_flow();
}

/// Demo 1: Context management - remembering what was discussed
fn demo_context_management() {
    println!("--- Demo 1: Context Management ---");
    println!("Chatbot maintains conversational context in toroidal memory\n");
    
    let mut bot = ToroidalChatbot::new(30);
    
    // Conversation about AI
    bot.process_message(
        "I'm interested in learning about machine learning",
        vec!["machine_learning", "AI", "learning"]
    );
    bot.generate_response();
    
    println!();
    
    // Related follow-up
    bot.process_message(
        "What about neural networks?",
        vec!["neural_networks", "AI", "machine_learning"]
    );
    bot.generate_response();
    
    bot.show_memory_state("Memory State - AI Discussion");
    
    println!("Key Advantage: Topics that co-occur frequently get spatially associated!");
    println!("'AI', 'machine_learning', and 'neural_networks' are now linked via diffusion.\n");
}

/// Demo 2: Topic association through proximity
fn demo_topic_association() {
    println!("\n--- Demo 2: Topic Association ---");
    println!("Related topics cluster together in memory space\n");
    
    let mut bot = ToroidalChatbot::new(25);
    
    // Discuss programming
    println!("Conversation 1 - Programming:");
    bot.process_message("Tell me about Rust", vec!["rust", "programming", "systems"]);
    bot.process_message("And Python?", vec!["python", "programming"]);
    bot.process_message("What about C++?", vec!["cpp", "programming", "systems"]);
    
    println!("\nConversation 2 - Cooking:");
    bot.process_message("I love Italian food", vec!["cooking", "italian", "food"]);
    bot.process_message("Especially pasta", vec!["cooking", "pasta", "italian"]);
    
    bot.show_memory_state("Memory - Two Topic Clusters");
    
    // Check associations
    println!("Topic Association Test:");
    println!("  Is 'rust' related to 'python'? {}", 
        bot.is_topic_related("rust", 0.2) && bot.is_topic_related("python", 0.2));
    println!("  Is 'rust' related to 'pasta'? {}", 
        bot.is_topic_related("rust", 0.2) && bot.is_topic_related("pasta", 0.2));
    
    println!("\nAdvantage: Topics associate naturally based on co-occurrence,");
    println!("without explicit relationship programming!\n");
}

/// Demo 3: Conversation flow and topic transitions
fn demo_conversation_flow() {
    println!("\n--- Demo 3: Conversation Flow & Memory Decay ---");
    println!("Demonstrating how old topics fade while recent ones stay active\n");
    
    let mut bot = ToroidalChatbot::new(30);
    
    // Long conversation with topic shifts
    println!("Turn 1:");
    bot.process_message("Let's talk about space exploration", vec!["space", "exploration"]);
    bot.generate_response();
    
    println!("\nTurn 2:");
    bot.process_message("What about Mars missions?", vec!["mars", "space", "exploration"]);
    bot.generate_response();
    
    // Let some time pass (memory decay)
    println!("\n[... some time passes - memory decays ...]");
    let engine = DiffusionEngine::new(DiffusionConfig {
        diffusion_rate: 0.1,
        decay_rate: 0.15,  // Higher decay
        threshold: 0.01,
    });
    engine.run(&mut bot.conversation_memory, 10);
    
    println!("\nTurn 3 (much later):");
    bot.process_message("Actually, I want to ask about cooking", vec!["cooking", "recipes"]);
    bot.generate_response();
    
    println!("\nActive topics after decay:");
    let active = bot.find_active_topics(5);
    for topic in active {
        let pos = bot.topic_positions.get(&topic).unwrap();
        let activation = bot.conversation_memory.get(pos.0, pos.1).unwrap();
        println!("  - {}: {:.3} activation", topic, activation);
    }
    
    println!("\nAdvantage: Natural forgetting! Old topics fade away,");
    println!("keeping the chatbot focused on recent context.\n");
}

// ============================================================================
// PRACTICAL ADVANTAGES OVER TRADITIONAL CHATBOT MEMORY
// ============================================================================

#[allow(dead_code)]
fn practical_advantages_explained() {
    println!("=== Why Toroidal Memory for Chatbots? ===\n");
    
    println!("1. CONTEXT WINDOW MANAGEMENT");
    println!("   Traditional: Fixed-size context buffer (e.g., last 10 messages)");
    println!("   Toroidal: Gradient of relevance - recent topics strong, old topics fade");
    println!("   Benefit: More natural 'forgetting' vs hard cutoff\n");
    
    println!("2. TOPIC ASSOCIATION");
    println!("   Traditional: Explicit knowledge graph or embedding similarity");
    println!("   Toroidal: Spatial proximity + diffusion = emergent associations");
    println!("   Benefit: Topics that co-occur naturally link together\n");
    
    println!("3. MULTI-TOPIC CONVERSATIONS");
    println!("   Traditional: Track topics in list, hard to handle multiple threads");
    println!("   Toroidal: Multiple peaks in activation landscape");
    println!("   Benefit: Can track parallel conversation threads spatially\n");
    
    println!("4. ATTENTION MECHANISM");
    println!("   Traditional: Attention weights computed each time");
    println!("   Toroidal: Attention spreads spatially from current focus");
    println!("   Benefit: More interpretable, can visualize what bot is 'thinking about'\n");
    
    println!("5. TEMPORAL DYNAMICS");
    println!("   Traditional: Static context representation");
    println!("   Toroidal: Dynamic - activation spreads and decays over time");
    println!("   Benefit: Natural model of working memory vs long-term memory\n");
    
    println!("6. GRACEFUL DEGRADATION");
    println!("   Traditional: Context limit hit = oldest messages dropped completely");
    println!("   Toroidal: Context fades gradually based on relevance");
    println!("   Benefit: Can still recall faded topics if re-activated\n");
}

// ============================================================================
// HYBRID APPROACH: Toroidal + Embeddings
// ============================================================================

/// Example of combining toroidal memory with semantic embeddings
#[allow(dead_code)]
struct HybridChatbot {
    spatial_memory: ToroidalMemory<f64>,
    semantic_embeddings: HashMap<String, Vec<f64>>,  // Topic -> embedding
    spatial_positions: HashMap<String, (isize, isize)>,
}

#[allow(dead_code)]
impl HybridChatbot {
    /// Best of both worlds approach:
    /// - Use embeddings to find semantic position (where to activate)
    /// - Use toroidal memory for temporal dynamics (how activation spreads)
    fn process_with_semantics(&mut self, topic: &str, embedding: Vec<f64>) {
        // 1. Use embedding to determine spatial position
        let position = self.embedding_to_position(&embedding);
        
        // 2. Activate in toroidal memory
        DiffusionEngine::activate_radius(
            &mut self.spatial_memory,
            position.0,
            position.1,
            2,
            1.0
        );
        
        // 3. Let it spread and interact with existing memory
        let engine = DiffusionEngine::with_defaults();
        engine.run(&mut self.spatial_memory, 5);
    }
    
    fn embedding_to_position(&self, embedding: &Vec<f64>) -> (isize, isize) {
        // Use dimensionality reduction (e.g., PCA) to map high-dim embedding to 2D
        // For now, simple hash
        let sum: f64 = embedding.iter().sum();
        let x = (sum * 13.7) as isize % 50;
        let y = (sum * 17.3) as isize % 50;
        (x.abs(), y.abs())
    }
}
