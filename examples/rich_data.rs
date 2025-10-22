/// Multi-Dimensional Data Storage in Toroidal Memory
/// 
/// This example demonstrates storing different types of data beyond just
/// activation levels: user facts, emotional states, preferences, and more.
/// 
/// Run with: cargo run --example rich_data

use ai_toroidal_memory::ToroidalMemory;
use std::collections::HashMap;

// ============================================================================
// APPROACH 1: Multiple Memory Layers (Different Data Types)
// ============================================================================

/// Multi-layer toroidal memory - like brain's different neural maps
struct MultiLayerMemory {
    // Layer 1: Activation/attention strength
    activation: ToroidalMemory<f64>,
    
    // Layer 2: Emotional valence (-1.0 to 1.0)
    emotion: ToroidalMemory<f64>,
    
    // Layer 3: Certainty/confidence (0.0 to 1.0)
    confidence: ToroidalMemory<f64>,
    
    // Layer 4: Recency (timestamp)
    recency: ToroidalMemory<f64>,
    
    // Metadata: What does each position represent?
    position_labels: HashMap<(isize, isize), String>,
}

impl MultiLayerMemory {
    fn new(size: usize) -> Self {
        MultiLayerMemory {
            activation: ToroidalMemory::new(size, size),
            emotion: ToroidalMemory::new(size, size),
            confidence: ToroidalMemory::new(size, size),
            recency: ToroidalMemory::new(size, size),
            position_labels: HashMap::new(),
        }
    }
    
    /// Store a rich memory with multiple attributes
    fn store_memory(
        &mut self,
        position: (isize, isize),
        label: &str,
        activation: f64,
        emotion: f64,      // -1.0 (negative) to 1.0 (positive)
        confidence: f64,   // 0.0 to 1.0
        timestamp: f64,    // Normalized time
    ) {
        self.activation.set(position.0, position.1, activation);
        self.emotion.set(position.0, position.1, emotion);
        self.confidence.set(position.0, position.1, confidence);
        self.recency.set(position.0, position.1, timestamp);
        self.position_labels.insert(position, label.to_string());
    }
    
    /// Query: Find all positive memories with high confidence
    fn find_positive_confident_memories(&self, threshold: f64) -> Vec<String> {
        let (width, height) = self.activation.dimensions();
        let mut results = Vec::new();
        
        for y in 0..height {
            for x in 0..width {
                let emotion = *self.emotion.get(x as isize, y as isize).unwrap();
                let confidence = *self.confidence.get(x as isize, y as isize).unwrap();
                
                if emotion > threshold && confidence > threshold {
                    if let Some(label) = self.position_labels.get(&(x as isize, y as isize)) {
                        results.push(label.clone());
                    }
                }
            }
        }
        
        results
    }
    
    /// Query: Get emotional context around a position
    fn get_emotional_context(&self, position: (isize, isize), radius: isize) -> f64 {
        let neighbors = self.emotion.get_neighbors(position.0, position.1, radius);
        let total: f64 = neighbors.iter().map(|(_, _, val)| *val).sum();
        total / neighbors.len() as f64
    }
    
    /// Weighted retrieval: Combine multiple factors
    fn compute_retrieval_score(&self, position: (isize, isize)) -> f64 {
        let activation = *self.activation.get(position.0, position.1).unwrap();
        let confidence = *self.confidence.get(position.0, position.1).unwrap();
        let recency = *self.recency.get(position.0, position.1).unwrap();
        
        // Weighted combination
        0.5 * activation + 0.3 * confidence + 0.2 * recency
    }
}

// ============================================================================
// APPROACH 2: Rich Data Structures (Enum-based)
// ============================================================================

/// Rich data that can be stored at each position
#[derive(Clone, Debug)]
enum MemoryContent {
    UserFact {
        fact: String,
        confidence: f64,
        last_mentioned: f64,
    },
    EmotionalState {
        emotion: String,        // "happy", "sad", "anxious"
        intensity: f64,
        trigger: Option<String>,
    },
    Preference {
        item: String,
        rating: f64,           // -1.0 to 1.0
        reasoning: String,
    },
    Event {
        description: String,
        timestamp: f64,
        importance: f64,
    },
    Relationship {
        entity: String,
        relationship_type: String,
        strength: f64,
    },
    Empty,
}

impl Default for MemoryContent {
    fn default() -> Self {
        MemoryContent::Empty
    }
}

/// Toroidal memory storing rich content
struct RichToroidalMemory {
    content: ToroidalMemory<MemoryContent>,
    activation: ToroidalMemory<f64>,  // Separate activation layer
}

impl RichToroidalMemory {
    fn new(size: usize) -> Self {
        RichToroidalMemory {
            content: ToroidalMemory::new(size, size),
            activation: ToroidalMemory::new(size, size),
        }
    }
    
    fn store_user_fact(&mut self, pos: (isize, isize), fact: String, confidence: f64) {
        let content = MemoryContent::UserFact {
            fact,
            confidence,
            last_mentioned: 1.0,
        };
        self.content.set(pos.0, pos.1, content);
        self.activation.set(pos.0, pos.1, confidence);
    }
    
    fn store_emotional_state(&mut self, pos: (isize, isize), emotion: String, intensity: f64) {
        let content = MemoryContent::EmotionalState {
            emotion,
            intensity,
            trigger: None,
        };
        self.content.set(pos.0, pos.1, content);
        self.activation.set(pos.0, pos.1, intensity);
    }
    
    fn query_by_type(&self, memory_type: &str) -> Vec<(isize, isize, MemoryContent)> {
        let (width, height) = self.content.dimensions();
        let mut results = Vec::new();
        
        for y in 0..height {
            for x in 0..width {
                let content = self.content.get(x as isize, y as isize).unwrap();
                
                let matches = match (memory_type, content) {
                    ("fact", MemoryContent::UserFact { .. }) => true,
                    ("emotion", MemoryContent::EmotionalState { .. }) => true,
                    ("preference", MemoryContent::Preference { .. }) => true,
                    _ => false,
                };
                
                if matches {
                    results.push((x as isize, y as isize, content.clone()));
                }
            }
        }
        
        results
    }
}

// ============================================================================
// PRACTICAL APPLICATIONS
// ============================================================================

fn main() {
    println!("=== Rich Data in Toroidal Memory ===\n");
    
    demo_emotional_tracking();
    demo_user_facts_network();
    demo_preference_mapping();
    demo_multi_modal_memory();
}

/// Demo 1: Track emotional states over conversation
fn demo_emotional_tracking() {
    println!("--- Demo 1: Emotional State Tracking ---");
    println!("Track user emotions throughout conversation (like RoBERTa sentiment)\n");
    
    let mut memory = MultiLayerMemory::new(30);
    
    // Simulate conversation with emotional analysis
    let conversation = vec![
        ("I love learning about AI!", 0.85, 10, 10),           // Positive
        ("This is frustrating", -0.6, 15, 15),                 // Negative
        ("But I'm making progress", 0.4, 12, 12),              // Mild positive
        ("I feel anxious about the deadline", -0.7, 20, 20),   // Anxious
        ("Accomplished something today!", 0.9, 8, 8),          // Very positive
    ];
    
    println!("Conversation with sentiment scores:");
    for (turn, (text, emotion, x, y)) in conversation.iter().enumerate() {
        memory.store_memory(
            (*x, *y),
            text,
            1.0,              // Full activation (recent)
            *emotion,         // Emotional valence
            0.8,              // Confidence in sentiment
            turn as f64,      // Recency
        );
        
        let emoji = if *emotion > 0.5 { "😊" } 
                    else if *emotion < -0.5 { "😟" } 
                    else { "😐" };
        
        println!("  Turn {}: {} {} (sentiment: {:.2})", turn + 1, emoji, text, emotion);
    }
    
    println!();
    
    // Analyze emotional patterns
    let positive_memories = memory.find_positive_confident_memories(0.5);
    println!("Positive experiences in conversation:");
    for mem in positive_memories {
        println!("  ✓ {}", mem);
    }
    
    // Check emotional context around anxiety
    let anxiety_context = memory.get_emotional_context((20, 20), 5);
    println!("\nEmotional context around anxiety point: {:.2}", anxiety_context);
    println!("(Negative value shows it's in a 'negative emotion cluster')\n");
}

/// Demo 2: User facts network with spatial relationships
fn demo_user_facts_network() {
    println!("--- Demo 2: User Facts Network ---");
    println!("Store facts about user with spatial relationships\n");
    
    let mut memory = MultiLayerMemory::new(40);
    
    // Store related facts near each other
    println!("Storing user facts:");
    
    // Work cluster (related facts close together)
    memory.store_memory((10, 10), "Works as software engineer", 1.0, 0.3, 0.9, 1.0);
    memory.store_memory((12, 11), "Uses Python daily", 0.9, 0.2, 0.85, 1.0);
    memory.store_memory((11, 13), "Interested in AI/ML", 1.0, 0.6, 0.9, 1.0);
    
    // Personal cluster (different region)
    memory.store_memory((30, 30), "Lives in San Francisco", 0.8, 0.0, 0.95, 1.0);
    memory.store_memory((32, 31), "Enjoys hiking", 0.7, 0.7, 0.8, 1.0);
    memory.store_memory((31, 33), "Has a dog named Max", 0.9, 0.8, 0.9, 1.0);
    
    println!("  Work-related facts clustered around (10, 10)");
    println!("  Personal facts clustered around (30, 30)");
    
    println!("\nBenefits of spatial organization:");
    println!("  1. Related facts activate together (spreading activation)");
    println!("  2. Can query by region ('Tell me about work' → activate work cluster)");
    println!("  3. Discover connections (facts stored nearby are related)");
    println!("  4. Visual debugging (see fact clusters in memory)\n");
}

/// Demo 3: Preference mapping
fn demo_preference_mapping() {
    println!("--- Demo 3: Preference Mapping ---");
    println!("Map user preferences spatially (like/dislike clusters)\n");
    
    let mut memory = MultiLayerMemory::new(35);
    
    // Foods - clustered by type
    println!("Food preferences:");
    memory.store_memory((5, 5), "Loves Italian food", 0.9, 0.9, 0.9, 1.0);
    memory.store_memory((7, 6), "Especially pasta", 0.8, 0.85, 0.85, 1.0);
    memory.store_memory((6, 8), "Favorite: carbonara", 0.95, 0.95, 0.9, 1.0);
    println!("  Italian food cluster: High positive emotion");
    
    memory.store_memory((25, 25), "Dislikes seafood", 0.7, -0.8, 0.85, 1.0);
    memory.store_memory((27, 26), "Allergic to shellfish", 0.9, -0.5, 0.95, 1.0);
    println!("  Seafood cluster: Negative emotion + high confidence");
    
    println!("\nWhat you can do:");
    println!("  1. Query: 'What does user like?' → Find high positive emotion regions");
    println!("  2. Avoid: 'Suggesting dinner' → Check negative emotion regions");
    println!("  3. Personalize: Recommend based on preference clusters");
    println!("  4. Explain: 'User likes Italian because...' → Associated facts nearby\n");
}

/// Demo 4: Multi-modal memory (combining different data types)
fn demo_multi_modal_memory() {
    println!("--- Demo 4: Multi-Modal Memory Integration ---");
    println!("Combine facts, emotions, and events in one memory\n");
    
    let mut rich_memory = RichToroidalMemory::new(30);
    
    // User mentions their job
    rich_memory.store_user_fact((10, 10), "Software engineer at TechCorp".to_string(), 0.9);
    rich_memory.store_emotional_state((10, 10), "pride".to_string(), 0.6);
    
    // User mentions feeling stressed
    rich_memory.store_emotional_state((15, 15), "stress".to_string(), 0.8);
    rich_memory.store_user_fact((16, 14), "Working on tight deadline".to_string(), 0.85);
    
    // User mentions their hobby
    rich_memory.store_user_fact((25, 25), "Plays guitar on weekends".to_string(), 0.8);
    rich_memory.store_emotional_state((25, 25), "joy".to_string(), 0.9);
    
    println!("Stored multi-modal memories:");
    println!("  Position (10, 10): Job fact + pride emotion");
    println!("  Position (15, 15): Stress emotion + deadline fact nearby");
    println!("  Position (25, 25): Hobby fact + joy emotion");
    
    println!("\nQuery by type:");
    let facts = rich_memory.query_by_type("fact");
    println!("  User facts stored: {}", facts.len());
    
    let emotions = rich_memory.query_by_type("emotion");
    println!("  Emotional states stored: {}", emotions.len());
    
    println!("\nPractical use:");
    println!("  - Bot can say: 'You mentioned feeling stressed about work deadlines'");
    println!("  - Combines fact (deadline) + emotion (stress) from nearby positions");
    println!("  - Can suggest: 'Maybe play some guitar to relax?' (from joy cluster)\n");
}

// ============================================================================
// INTEGRATION WITH SENTIMENT ANALYSIS (RoBERTa-style)
// ============================================================================

#[allow(dead_code)]
mod sentiment_integration {
    use super::*;
    
    /// Mock sentiment analysis (in real app, use transformers/RoBERTa)
    struct SentimentAnalyzer;
    
    impl SentimentAnalyzer {
        fn analyze(text: &str) -> EmotionalAnalysis {
            // In production: Call RoBERTa or similar model
            // For demo: Simple keyword matching
            
            let text_lower = text.to_lowercase();
            
            let (emotion, intensity) = if text_lower.contains("love") || text_lower.contains("great") {
                ("joy", 0.8)
            } else if text_lower.contains("anxious") || text_lower.contains("worried") {
                ("anxiety", 0.7)
            } else if text_lower.contains("frustrated") || text_lower.contains("angry") {
                ("anger", 0.6)
            } else if text_lower.contains("sad") || text_lower.contains("depressed") {
                ("sadness", 0.7)
            } else {
                ("neutral", 0.1)
            };
            
            EmotionalAnalysis {
                primary_emotion: emotion.to_string(),
                intensity,
                valence: if matches!(emotion, "joy") { 0.8 } else { -0.6 },
                arousal: intensity,
            }
        }
    }
    
    #[derive(Debug)]
    struct EmotionalAnalysis {
        primary_emotion: String,
        intensity: f64,
        valence: f64,    // -1.0 (negative) to 1.0 (positive)
        arousal: f64,    // 0.0 (calm) to 1.0 (excited)
    }
    
    /// Bot with sentiment-aware memory
    pub struct SentimentAwareBot {
        memory: MultiLayerMemory,
        turn_count: usize,
    }
    
    impl SentimentAwareBot {
        pub fn new() -> Self {
            SentimentAwareBot {
                memory: MultiLayerMemory::new(40),
                turn_count: 0,
            }
        }
        
        pub fn process_message(&mut self, message: &str) {
            // Analyze sentiment
            let sentiment = SentimentAnalyzer::analyze(message);
            
            // Map to spatial position (could use topic modeling + embeddings)
            let pos = self.hash_to_position(message);
            
            // Store with emotional data
            self.memory.store_memory(
                pos,
                message,
                1.0,                    // Full activation (recent)
                sentiment.valence,      // Emotional valence
                0.8,                    // Confidence
                self.turn_count as f64, // Recency
            );
            
            self.turn_count += 1;
            
            println!("Stored: '{}' with emotion: {} (valence: {:.2})",
                message, sentiment.primary_emotion, sentiment.valence);
        }
        
        fn hash_to_position(&self, text: &str) -> (isize, isize) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let hash = hasher.finish();
            
            let x = (hash % 40) as isize;
            let y = ((hash / 40) % 40) as isize;
            (x, y)
        }
        
        pub fn get_emotional_summary(&self) -> String {
            let positive = self.memory.find_positive_confident_memories(0.5);
            format!("Overall emotional tone: {} positive moments detected", positive.len())
        }
    }
}

// ============================================================================
// WHAT YOU CAN DO WITH RICH DATA
// ============================================================================

#[allow(dead_code)]
fn applications_of_rich_data() {
    println!("\n=== Applications of Rich Data in Toroidal Memory ===\n");
    
    println!("1. EMOTIONAL INTELLIGENCE");
    println!("   - Track user's emotional journey");
    println!("   - Detect mood shifts (clustering of negative emotions)");
    println!("   - Empathetic responses based on emotional context");
    println!("   - Example: 'I notice you seem stressed about work lately'\n");
    
    println!("2. PERSONALIZATION");
    println!("   - Learn preferences spatially");
    println!("   - Cluster: likes/dislikes");
    println!("   - Recommend based on positive emotion regions");
    println!("   - Example: 'Based on your love of Italian food, try this restaurant'\n");
    
    println!("3. THERAPEUTIC APPLICATIONS");
    println!("   - Mental health tracking");
    println!("   - Identify negative thought patterns (clusters)");
    println!("   - Track progress over time (emotional regions changing)");
    println!("   - Example: 'Your anxiety about work has decreased this week'\n");
    
    println!("4. CONTEXT-AWARE REASONING");
    println!("   - Combine facts + emotions + events");
    println!("   - Understand WHY user feels certain way");
    println!("   - Example: stress (emotion) + deadline (fact) → empathetic response\n");
    
    println!("5. CONTRADICTION DETECTION");
    println!("   - Same topic, different confidence levels");
    println!("   - User says conflicting things → different positions");
    println!("   - Example: 'Earlier you said X, now Y - let's clarify'\n");
    
    println!("6. INTEREST MAPPING");
    println!("   - Topics with high activation = interests");
    println!("   - Spatial clusters = related interests");
    println!("   - Guide conversation toward interest regions\n");
    
    println!("7. MEMORY REPLAY");
    println!("   - 'Remember when you mentioned...'");
    println!("   - Activate old memory region");
    println!("   - Associated facts/emotions resurface\n");
    
    println!("8. RELATIONSHIP GRAPHS");
    println!("   - People/entities at different positions");
    println!("   - Distance = relationship strength");
    println!("   - Emotional valence per relationship");
    println!("   - Example: 'You seem happy when you talk about your sister'\n");
}
