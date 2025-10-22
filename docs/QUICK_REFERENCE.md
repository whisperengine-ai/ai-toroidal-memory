# Quick Reference: Toroidal Memory for AI

> 📚 **Documentation:** [README](../README.md) | [Plain English Guide](PLAIN_ENGLISH_GUIDE.md) | [Terminal Chat](TERMINAL_CHAT.md) | [API Docs](API_DOCUMENTATION.md) | [Concepts](CONCEPTS.md)

## TL;DR

**Toroidal Memory** = 2D grid where edges wrap around (like a donut surface)

```
┌─────────┐
│    →    │  Left edge connects to right edge
│  ↓ ↑    │  Top edge connects to bottom edge  
│    ←    │  = No boundaries, continuous space
└─────────┘
```

---

## Core Concept in 3 Points

1. **Space-based memory**: Instead of storing memories in a list or database, store them as activation patterns in 2D space
2. **No boundaries**: Toroidal topology means every location is treated equally (no edge cases!)
3. **Spreading activation**: Memories influence nearby regions through diffusion (like ripples in water)

---

## How Memories Are Created

### 1. Position Encoding
```rust
// Map concept → spatial position
let (x, y) = hash("apple");  // e.g., (15, 20)
memory.set(x, y, activation);
```

### 2. Activation Spreading
```rust
// Activate position and let it spread
DiffusionEngine::activate_radius(&mut memory, x, y, radius, strength);
engine.run(&mut memory, steps);  // Spreads over time
```

### 3. Association Formation
```rust
// Co-activate related concepts
activate(&mut memory, concept_A_position);
activate(&mut memory, concept_B_position);
// Diffusion creates gradient path between them
```

---

## Key Properties

| Property | Traditional AI Memory | Toroidal Memory |
|----------|----------------------|-----------------|
| Structure | Lists, vectors, graphs | 2D spatial grid |
| Boundaries | Yes (edges) | No (wraps around) |
| Retrieval | Exact key/similarity search | Spatial proximity + diffusion |
| Dynamics | Static | Temporal (diffusion over time) |
| Associations | Explicit links | Emergent from proximity |
| Biological | Abstract | Maps to grid cells in brain! |

---

## Comparisons

### vs Vector Databases (FAISS, Pinecone)
- **Vector DB**: High-dimensional embeddings, discrete items, semantic search
- **Toroidal**: 2D spatial structure, continuous activation, emergent associations
- **Use Toroidal when**: You need spatial reasoning, temporal dynamics, or embodied AI

### vs Transformer Attention
- **Transformer**: Content-based, parallel, proven for sequences
- **Toroidal**: Position-based, diffusion dynamics, spatial topology
- **Use Toroidal when**: Processing spatial data, need explicit locality, want interpretable memory

### vs Neural Memory Networks (NTM, DNC)
- **NTM/DNC**: Learnable addressing, algorithmic tasks, complex architecture
- **Toroidal**: Fixed topology, spatial tasks, simple and interpretable
- **Use Toroidal when**: Research, visualization needs, spatial embodiment

### Most Similar To: Grid Cells in Neuroscience!
- Brain actually uses toroidal-like representations for spatial memory
- Grid cells fire in periodic patterns across space
- Multiple scales create unique "GPS" for positions
- **Toroidal memory is biologically inspired!**

---

## When to Use Toroidal Memory

### ✅ Great For:
- **Robot navigation**: Map environment without edge artifacts
- **Spatial reasoning**: Agent needs to reason about locations
- **Embodied AI**: Physical agents in space
- **Procedural generation**: Tile patterns seamlessly
- **Research**: Explore spatial memory models
- **Visualization**: Can literally see memory states!

### ❌ Not Ideal For:
- **Pure language tasks**: No inherent spatial structure
- **Large-scale search**: Fixed size, not as scalable as vector DBs
- **Exact retrieval**: Better for fuzzy, associative recall
- **Already have working solution**: Don't fix what isn't broken

---

## Example Use Case: Robot Memory

```rust
// Robot explores environment
let mut spatial_memory = ToroidalMemory::new(100, 100);

// Stores observations at locations
for observation in sensor_data {
    let (x, y) = observation.position;
    DiffusionEngine::activate_radius(&mut spatial_memory, x, y, 3, 1.0);
}

// Consolidate memories over time
let engine = DiffusionEngine::with_defaults();
engine.run(&mut spatial_memory, 10);

// Query: "What's at position (50, 50)?"
let activation = spatial_memory.get(50, 50).unwrap();
if activation > threshold {
    println!("Robot has been here before!");
}

// Query: "Find places with high activity"
let hotspots = find_peaks(&spatial_memory);
```

---

## The "Aha!" Moment

**Traditional Memory:**
```
"apple" → [0.1, 0.9, 0.3, ...] → Store in vector DB → Retrieve by similarity
         ^embedding (abstract)
```

**Toroidal Memory:**
```
"apple" → Position (15, 20) → Activate region → Spreads to nearby concepts
         ^spatial location (concrete)
```

The key insight: **Space itself encodes relationships!**
- Close in space = related concepts
- Diffusion = information spreading
- No edges = consistent behavior everywhere

---

## Running the Examples

```bash
# Basic demo
cargo run

# Advanced examples (cellular automata, attention)
cargo run --example advanced

# Memory formation (episodic, associative, interference)
cargo run --example memory_formation

# Chatbot with toroidal memory
cargo run --example chatbot

# State persistence (binary, JSON, compressed)
cargo run --example persistence

# Rich data: emotions, facts, preferences
cargo run --example rich_data

# GPU acceleration and LLM integration
cargo run --example gpu_and_llm

# Run tests
cargo test
```

---

## Emotion Scoring Quick Guide

### Storage Format
```rust
// Multi-layer memory with emotions
struct EmotionalMemory {
    activation: ToroidalMemory<f64>,  // 0.0 to 1.0 (importance)
    emotion: ToroidalMemory<f64>,     // -1.0 to 1.0 (valence)
    confidence: ToroidalMemory<f64>,  // 0.0 to 1.0 (certainty)
}
```

### Valence Scale
```
-1.0 = Very negative (sad, angry, anxious)
 0.0 = Neutral
+1.0 = Very positive (happy, joyful, excited)
```

### Integration Pattern
```rust
// 1. Analyze sentiment (RoBERTa or similar)
let sentiment = analyze_text(user_message);

// 2. Store with emotion
memory.store(
    position,
    activation: 1.0,
    emotion: sentiment.valence,  // -1.0 to +1.0
    confidence: sentiment.score,
);

// 3. Generate LLM context
let positive_topics = memory.find_positive_memories(0.5);
let concerns = memory.find_negative_memories(-0.5);
let prompt = generate_prompt(positive_topics, concerns);
```

### Use Cases
- 🤖 **Chatbots**: Track user sentiment, respond empathetically
- 🧠 **Mental Health**: Monitor emotional patterns over time
- 💬 **Customer Service**: Detect frustration, escalate appropriately
- 📚 **Education**: Track student stress and engagement
- 🎮 **Games**: NPCs remember emotional interactions

---

## Further Reading

- See `docs/CONCEPTS.md` for deep dive (including emotion scoring details)
- See `docs/CHATBOT_APPLICATION.md` for conversational AI specifics
- See `docs/STATE_MANAGEMENT.md` for persistence patterns
- See `docs/PROJECT_SUMMARY.md` for complete feature overview
- See code comments in `src/` for implementation details
- Research papers on grid cells and spatial cognition
- Neural Cellular Automata research (Mordvintsev et al.)

---

## Bottom Line

Toroidal memory offers a **spatial, continuous, boundary-free** alternative to traditional AI memory structures. It's especially powerful for:
- Embodied AI and robotics
- Spatial reasoning tasks  
- Research into biologically-inspired architectures
- Any application where spatial relationships matter
- **Conversational AI with emotional intelligence**
- **Chatbots that remember context and sentiment**

Think of it as giving your AI a "mental map" where memories have locations and can influence each other through proximity. Add emotional layers and you get an AI that understands not just *what* was said, but *how* the user felt. 🗺️🧠💙

