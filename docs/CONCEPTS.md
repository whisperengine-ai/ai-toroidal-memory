# Toroidal Memory for AI: Deep Dive

> 📚 **Documentation:** [README](../README.md) | [Plain English Guide](PLAIN_ENGLISH_GUIDE.md) | [Quick Reference](QUICK_REFERENCE.md) | [Terminal Chat](TERMINAL_CHAT.md) | [AI Comparison](AI_MEMORY_COMPARISON.md)

## What is Toroidal Memory?

### The Torus Shape

A **torus** is a donut-shaped 3D surface. In 2D, a toroidal space is like a rectangle where:
- The left edge connects to the right edge (wrap horizontally)
- The top edge connects to the bottom edge (wrap vertically)

```
Traditional Grid:          Toroidal Grid:
┌─────────┐               ┌─────────┐
│  edges  │               │ wraps   │
│  stop   │    vs         │ around  │  (imagine left→right, top→bottom connected)
│  here   │               │ edges   │
└─────────┘               └─────────┘
```

### Why "Toroidal" for AI Memory?

In traditional memory structures, edges create special cases:
- **Boundary problems**: Different behavior at edges vs center
- **Discontinuities**: Artificial breaks in spatial relationships
- **Edge artifacts**: Patterns near boundaries behave differently

Toroidal memory eliminates these issues by creating a **continuous, homogeneous space** where every position is equivalent.

---

## How This is Used for AI Memory

### 1. **Spatial Memory Representation**

AI agents often need to represent spatial information:

```rust
// Example: Robot mapping an environment
let mut world_memory = ToroidalMemory::new(100, 100);

// Store what the agent sees at each location
world_memory.set(x, y, Observation {
    obstacle: true,
    explored: true,
    reward: 0.5,
});

// Query neighbors - works seamlessly even at edges!
let neighbors = world_memory.get_neighbors(x, y, radius);
```

**Advantages for AI:**
- No special-case code for boundaries
- Consistent spatial queries everywhere
- Natural for procedurally infinite worlds (tiling pattern)

### 2. **Pattern Storage and Recognition**

Memories can be stored as activation patterns:

```rust
// Store a memory as an activation pattern
let mut memory = ToroidalMemory::new(50, 50);

// When agent experiences something (e.g., sees a face)
DiffusionEngine::activate_radius(&mut memory, 25, 25, 5, 1.0);

// Let it settle through diffusion (creates unique signature)
let engine = DiffusionEngine::with_defaults();
engine.run(&mut memory, 10);
```

The diffusion creates a **unique spatial signature** that can later be recognized.

### 3. **Associative Memory Networks**

Toroidal memory enables spreading activation models (like neural networks):

```
Input stimulus → Activates location → Spreads to associated memories → Retrieval
```

```rust
// Activate based on input features
for feature in detected_features {
    let (x, y) = hash_feature_to_position(feature);
    DiffusionEngine::activate(&mut memory, x, y, feature.strength);
}

// Let activation spread
engine.run(&mut memory, steps);

// Read out strongest activations as recalled memories
let recalled = find_peaks(&memory);
```

---

## How New Memories Are Created

### Method 1: Direct Encoding

```rust
// Store explicit memory at a location
let memory_location = hash(memory_content); // deterministic position
memory.set(location.x, location.y, memory_content);
```

### Method 2: Hebbian-like Learning

"Neurons that fire together, wire together"

```rust
struct MemorySystem {
    activation: ToroidalMemory<f64>,
    weights: ToroidalMemory<f64>,
}

impl MemorySystem {
    fn learn(&mut self, pattern: &Pattern) {
        // 1. Activate pattern
        self.activate_pattern(pattern);
        
        // 2. Run diffusion (temporal aspect)
        self.diffusion_step();
        
        // 3. Strengthen connections where activation co-occurs
        for x in 0..self.width {
            for y in 0..self.height {
                let activation = self.activation.get(x, y).unwrap();
                let neighbors = self.activation.get_neighbors(x, y, 1);
                
                // Hebbian rule: strengthen based on co-activation
                for (nx, ny, neighbor_activation) in neighbors {
                    let weight_increase = activation * neighbor_activation * learning_rate;
                    // Update weights between positions
                }
            }
        }
    }
}
```

### Method 3: Competitive Learning

Multiple patterns compete for representation space:

```rust
fn store_memory(memory: &mut ToroidalMemory<f64>, pattern: Vec<f64>) {
    // Find location with best match (or least interference)
    let best_location = find_best_storage_location(memory, &pattern);
    
    // Store with some overlap tolerance
    encode_pattern_at(memory, &pattern, best_location);
}

fn find_best_storage_location(
    memory: &ToroidalMemory<f64>, 
    pattern: &Vec<f64>
) -> (usize, usize) {
    // Find "empty" or most similar location
    // Balances: similarity (for association) vs interference (for separation)
    // ...
}
```

---

## Comparison to Other AI Memory Approaches

### 1. **vs Vector Databases (Traditional Embedding Memory)**

#### Vector Databases (e.g., FAISS, Pinecone)
```
Input → Embedding Model → Vector → Store in high-dim space → Nearest neighbor search
```

**Pros:**
- Semantic similarity through embeddings
- Scalable to millions of items
- Well-established retrieval methods

**Cons:**
- No spatial structure
- No temporal dynamics
- Discrete, non-continuous
- No spreading activation

#### Toroidal Memory

**Pros:**
- Continuous spatial representation
- Natural temporal dynamics (diffusion)
- Emergent pattern interactions
- Boundary-free processing
- Supports spreading activation

**Cons:**
- Fixed size (though can be tiled)
- Not inherently semantic (needs encoding)
- Less mature tooling

**Best for:** Spatial reasoning, embodied AI, real-time dynamic systems

---

### 2. **vs Transformer Attention Memory**

#### Transformer Attention
```
Query × Key^T → Attention weights → Weighted sum of Values
```

**Pros:**
- Content-based addressing
- Parallel computation
- State-of-the-art for sequences
- Flexible context

**Cons:**
- Quadratic complexity (O(n²))
- No inherent spatial structure
- Discrete positions
- Limited temporal dynamics

#### Toroidal Memory

**Pros:**
- Fixed complexity per step
- Natural spatial topology
- Continuous diffusion dynamics
- Implicit temporal evolution

**Cons:**
- No content-based addressing (without additional mechanism)
- Fixed spatial dimensions
- Not proven on language tasks

**Potential Hybrid:** Toroidal attention layer where attention spreads spatially

---

### 3. **vs Working Memory Networks (DNC, NTM)**

#### Differentiable Neural Computers
```
Controller → Read/Write heads → External memory matrix → Content + Location addressing
```

**Pros:**
- Learnable read/write operations
- Content and location-based access
- Proven on algorithmic tasks
- Differentiable (end-to-end training)

**Cons:**
- Complex architecture
- Difficult to train
- Memory as flat matrix
- No spatial topology

#### Toroidal Memory

**Pros:**
- Simple, interpretable
- Built-in spatial relationships
- Natural diffusion dynamics
- Easy to visualize

**Cons:**
- Not differentiable by default (but could be)
- No learned addressing
- Requires manual encoding strategy

**Use cases:**
- DNC: Learning algorithms, reasoning
- Toroidal: Spatial navigation, embodied cognition

---

### 4. **vs Hopfield Networks / Associative Memory**

#### Hopfield Networks
```
Store patterns as weight matrix → Energy minimization → Pattern retrieval
```

**Similarities:**
- Both are attractor-based
- Pattern completion
- Associative retrieval

**Differences:**

| Aspect | Hopfield | Toroidal |
|--------|----------|----------|
| Topology | Fully connected | Spatial 2D grid |
| Dynamics | Energy descent | Diffusion process |
| Capacity | ~0.15N patterns | Depends on spatial encoding |
| Retrieval | Global convergence | Local spreading activation |
| Structure | Abstract | Spatially explicit |

**Toroidal advantages:**
- Explicit spatial relationships
- Local operations (more biologically plausible)
- Continuous dynamics
- Composable patterns

---

### 5. **vs Hippocampal-inspired Models (Grid Cells, Place Cells)**

#### Neuroscience Inspiration

The brain uses:
- **Place cells**: Fire at specific locations
- **Grid cells**: Fire in hexagonal grid patterns (toroidal-like!)
- **Head direction cells**: Encode orientation

#### Toroidal Memory Connection

Toroidal memory is actually **quite similar** to grid cell models:

```rust
// Grid cells often modeled with toroidal topology!
// Multiple toroidal memories at different scales

struct GridCellMemory {
    scales: Vec<ToroidalMemory<f64>>,  // Different spatial scales
    phases: Vec<(f64, f64)>,            // Phase offsets
}

impl GridCellMemory {
    fn encode_position(&mut self, x: f64, y: f64) {
        for (i, grid) in self.scales.iter_mut().enumerate() {
            let scale = 2.0_f64.powi(i as i32);
            let gx = (x / scale + self.phases[i].0) as isize;
            let gy = (y / scale + self.phases[i].1) as isize;
            
            DiffusionEngine::activate_radius(grid, gx, gy, 2, 1.0);
        }
    }
    
    fn decode_position(&self) -> (f64, f64) {
        // Combine activations across scales to determine position
        // (This is how grid cells work in the brain!)
    }
}
```

**This is perhaps the MOST biologically plausible approach!**

---

## Practical Use Cases for Toroidal AI Memory

### 1. **Robot Navigation**
```rust
// Store spatial map of environment
// No edge artifacts when exploring
// Natural wrap-around for cyclic environments (circular tracks, etc.)
```

### 2. **Procedural Generation**
```rust
// Generate infinite worlds by tiling toroidal patterns
// Seamless boundaries
// Use for terrain, textures, game levels
```

### 3. **Attention-based Vision**
```rust
// Image features activate memory regions
// Attention spreads to related features
// Top-down feedback influences perception
```

### 4. **Reinforcement Learning State Space**
```rust
// States encoded as positions in toroidal memory
// Value function spreads through similar states
// No boundary bias in policy learning
```

### 5. **Neural Cellular Automata**
```rust
// Toroidal grid for pattern generation
// Used in recent AI research for:
//   - Texture synthesis
//   - Morphogenesis
//   - Self-organizing systems
```

---

## Advanced Concepts

### Multi-Scale Toroidal Memory

```rust
struct HierarchicalMemory {
    fine: ToroidalMemory<f64>,      // 100x100 - details
    medium: ToroidalMemory<f64>,    // 50x50 - regions  
    coarse: ToroidalMemory<f64>,    // 25x25 - global patterns
}

// Bottom-up: Pool fine activations to coarse levels
// Top-down: Bias fine activations from coarse predictions
```

### Temporal Binding

```rust
// Store sequences as trails through toroidal space
struct SequenceMemory {
    space: ToroidalMemory<f64>,
    trajectory: Vec<(isize, isize)>,  // Path through space
}

// Retrieve by following activation gradient
```

### Hybrid Approaches

```rust
// Toroidal memory + vector embeddings
struct HybridMemory {
    spatial: ToroidalMemory<f64>,           // Spatial structure
    embeddings: HashMap<(isize, isize), Vec<f64>>,  // Semantic content
}

// Position encodes spatial relationships
// Embeddings encode semantic content
// Best of both worlds!
```

---

## Research Directions

1. **Learning Spatial Encodings**: How to automatically learn good mappings from concepts to spatial positions?

2. **Differentiable Toroidal Memory**: Make it trainable via backpropagation for deep learning integration

3. **Optimal Diffusion Parameters**: What decay/spread rates maximize memory capacity?

4. **Toroidal Transformers**: Replace linear attention with spatially-structured toroidal attention

5. **Multi-Modal Binding**: How to store text, images, and actions in unified toroidal space?

6. **Catastrophic Forgetting**: Does spatial separation help prevent overwriting old memories?

---

## Emotion Scoring and Sentiment Integration

### Multi-Dimensional Emotional Memory

Modern AI chatbots benefit from tracking emotional context alongside factual information. Toroidal memory naturally supports multi-layer storage:

```rust
struct EmotionalMemory {
    activation: ToroidalMemory<f64>,  // Attention/importance
    emotion: ToroidalMemory<f64>,     // Valence (-1.0 to 1.0)
    confidence: ToroidalMemory<f64>,  // Certainty (0.0 to 1.0)
    recency: ToroidalMemory<f64>,     // Temporal info
}
```

### Emotion Scoring Scale

**Valence Spectrum** (-1.0 to 1.0):
```
-1.0 ◄─────────────┼─────────────► +1.0
Negative        Neutral        Positive

-1.0 to -0.7: Strong negative (sadness, depression, anger)
-0.7 to -0.3: Mild negative (worry, frustration, disappointment)
-0.3 to +0.3: Neutral (factual statements, calm observations)
+0.3 to +0.7: Mild positive (contentment, interest, hope)
+0.7 to +1.0: Strong positive (joy, excitement, love)
```

### RoBERTa-Style Sentiment Analysis

**Production Integration Pattern**:

```rust
use transformers::{RobertaModel, Tokenizer};

struct SentimentAnalyzer {
    model: RobertaModel,
    tokenizer: Tokenizer,
}

impl SentimentAnalyzer {
    fn analyze(&self, text: &str) -> EmotionalAnalysis {
        // Tokenize and run through RoBERTa
        let tokens = self.tokenizer.encode(text);
        let output = self.model.forward(tokens);
        
        // Get emotion classification
        let scores = softmax(output);
        
        EmotionalAnalysis {
            primary_emotion: get_top_emotion(scores), // "joy", "sadness", etc.
            intensity: scores.max(),                   // 0.0 - 1.0
            valence: emotion_to_valence(scores),       // -1.0 to 1.0
            arousal: calculate_arousal(scores),        // 0.0 - 1.0 (calm/excited)
        }
    }
}
```

**Emotion Categories**:
- **Joy** (valence: +0.8 to +1.0): Happiness, excitement, delight
- **Love** (valence: +0.7 to +0.9): Affection, care, warmth
- **Surprise** (valence: +0.3 to +0.5): Unexpected positive events
- **Neutral** (valence: -0.2 to +0.2): Factual, informational
- **Sadness** (valence: -0.7 to -0.9): Grief, disappointment
- **Anger** (valence: -0.6 to -0.8): Frustration, rage
- **Fear/Anxiety** (valence: -0.5 to -0.8): Worry, stress, concern

### Spatial Emotional Clustering

Emotions stored in toroidal space naturally cluster:

```
Positive Region (Top-Left):
  ┌─────────────┐
  │ 😊 hobbies  │
  │ 😍 family   │
  │ 🎉 achievements│
  └─────────────┘

Negative Region (Bottom-Right):
  ┌─────────────┐
  │ 😰 work stress│
  │ 😟 deadlines │
  │ 😢 conflicts │
  └─────────────┘
```

**Benefits**:
1. **Pattern Detection**: Identify recurring emotional themes
2. **Context Awareness**: LLM sees emotional state around topics
3. **Empathetic Responses**: React appropriately to user's mood
4. **Trend Analysis**: Track emotional journey over time
5. **Intervention Points**: Detect concerning negative clusters

### Integration with LLM Prompts

Convert emotional memory to rich context:

```rust
fn generate_emotional_prompt(memory: &EmotionalMemory) -> String {
    let positive = memory.find_regions(|emotion| emotion > 0.5);
    let negative = memory.find_regions(|emotion| emotion < -0.5);
    
    format!(
        "User's emotional context:\n\
         Positive topics: {}\n\
         Concerns: {}\n\
         Be empathetic and reference relevant topics.",
        format_topics(positive),
        format_topics(negative)
    )
}
```

### Use Cases

1. **Mental Health Chatbots**: Track mood patterns, detect decline
2. **Customer Service**: Gauge satisfaction, escalate frustrated users
3. **Education**: Monitor student engagement and stress
4. **Therapy Assistants**: Identify cognitive patterns, track progress
5. **Social Robots**: Natural emotional responses to human interaction

### Implementation Options

**Option 1: External Sentiment API**
```rust
// Call to HuggingFace, OpenAI, etc.
let response = reqwest::post("https://api-inference.huggingface.co/models/...")
    .json(&json!({"inputs": text}))
    .send().await?;
```

**Option 2: Local ONNX Model**
```rust
// Load RoBERTa model locally
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};
let model = session_builder.with_model_from_file("roberta-sentiment.onnx")?;
```

**Option 3: Simple Heuristics** (for demos)
```rust
// Keyword-based approximation
fn simple_sentiment(text: &str) -> f64 {
    let positive_words = ["love", "great", "happy", "excellent", "wonderful"];
    let negative_words = ["hate", "terrible", "awful", "frustrated", "sad"];
    
    // Count and score
    let pos_count = positive_words.iter().filter(|w| text.contains(w)).count();
    let neg_count = negative_words.iter().filter(|w| text.contains(w)).count();
    
    (pos_count as f64 - neg_count as f64) / (pos_count + neg_count + 1) as f64
}
```

---

## Summary

**Toroidal Memory is:**
- ✅ Spatially structured (unlike vectors)
- ✅ Boundary-free (unlike grids)
- ✅ Dynamic (unlike static databases)
- ✅ Biologically inspired (grid cells!)
- ✅ Composable (multiple scales, modalities)
- ✅ Emotionally aware (multi-layer support)
- ✅ LLM-ready (rich prompt generation)

**Best suited for:**
- Embodied AI agents
- Spatial reasoning tasks
- Real-time dynamic systems
- Visualization of memory states
- Research into spatial cognition
- Emotionally intelligent chatbots
- Conversational AI with context
- Mental health and therapy applications

**Challenges:**
- Encoding strategy design
- Scaling to very large memories
- Integration with modern deep learning
- Limited existing research/tooling
- Sentiment model selection and integration

This project provides a foundation for exploring these ideas in Rust! 🚀

