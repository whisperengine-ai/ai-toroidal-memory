# AI Toroidal Memory

An exploration of toroidal (torus/donut-shaped) memory structures for AI applications. This project implements a 2D toroidal memory space where edges wrap around, creating a continuous surface without boundaries.

> 💡 **New to toroidal memory?** Start with [Plain English Guide](docs/PLAIN_ENGLISH_GUIDE.md) or [Quick Reference](docs/QUICK_REFERENCE.md)  
> 🧠 **Want deep explanations?** Read [Concepts Guide](docs/CONCEPTS.md)  
> 🚀 **Try it now:** `cargo run --example terminal_chat` for an interactive chatbot!  
> 🐳 **Docker deployment:** `docker-compose up -d` for instant API server!

## What is This?

Toroidal memory is a spatial memory model where:
- Memories are stored as activation patterns in 2D space
- Space wraps around (like a donut/torus) - no boundaries!
- Activation spreads through diffusion (like neural networks)
- Similar to how the brain uses grid cells for spatial memory

**Use cases:** Robot navigation, spatial reasoning, embodied AI, procedural generation, neuroscience research, **chatbots with persistent context**

## Concepts

### Toroidal Topology
A torus is a surface where:
- The left edge connects to the right edge
- The top edge connects to the bottom edge
- No point is at a boundary or edge

This creates interesting properties for:
- **Spatial reasoning**: Continuous space without edge cases
- **Pattern recognition**: Patterns can wrap around edges
- **Neural models**: Activation spreading without boundaries
- **Cellular automata**: No special boundary conditions needed

## Features

### 1. **Toroidal Memory** (`toroidal_memory.rs`)
Core data structure with:
- Automatic coordinate wrapping
- Neighbor queries
- Generic over data types
- Efficient indexing

### 2. **Pattern Matching** (`pattern_matcher.rs`)
Pattern recognition capabilities:
- Find patterns anywhere in memory (including wrapped)
- Similarity scoring
- Multiple pattern instances

### 3. **Diffusion Engine** (`diffusion.rs`)
Neural-like activation spreading:
- Diffusion across the toroidal surface
- Configurable decay and spread rates
- Localized and radius-based activation
- **Multi-core parallel processing** (2-8x speedup on large grids)
- Configurable decay and spread rates
- Localized and radius-based activation

### 4. **Multi-Layer Memory** (Examples)
Rich data structures for AI:
- Emotional valence tracking (-1.0 to 1.0)
- Confidence/certainty layers (0.0 to 1.0)
- User facts and preferences
- Temporal/recency information

### 5. **Sentiment Integration** (Examples)
Emotion scoring with RoBERTa-style analysis:
- Primary emotion detection (joy, anxiety, anger, sadness)
- Intensity and valence scoring
- Spatial clustering of emotional states
- Multi-modal memory (facts + emotions)

## Usage

### Basic Memory Operations

```rust
use toroidal_memory::ToroidalMemory;

// Create a 10x10 toroidal memory
let mut memory = ToroidalMemory::new(10, 10);

// Set and get values (coordinates wrap automatically)
memory.set(0, 0, 42);
let value = memory.get(-10, -10); // Same as (0, 0)!

// Get neighbors
let neighbors = memory.get_neighbors(5, 5, 1);
```

### Pattern Matching

```rust
use pattern_matcher::{Pattern, PatternMatcher};

// Define a pattern
let pattern = Pattern::new(2, 2, vec![1, 2, 3, 4]);

// Find all occurrences
let matches = PatternMatcher::find_pattern(&memory, &pattern);

// Calculate similarity
let score = PatternMatcher::similarity_score(&memory, &pattern, 0, 0);
```

### Diffusion Simulation

```rust
use diffusion::{DiffusionEngine, DiffusionConfig};

let mut memory = ToroidalMemory::new(20, 20);

// Activate a region
DiffusionEngine::activate_radius(&mut memory, 10, 10, 3, 1.0);

// Run diffusion
let engine = DiffusionEngine::with_defaults();
engine.run(&mut memory, 10);
```

## Running the Examples

```bash
# Basic demo - shows core features
cargo run

# Advanced examples - cellular automata, attention spreading
cargo run --example advanced

# Memory formation - how memories are created and associated
cargo run --example memory_formation

# Chatbot - practical conversational AI with toroidal memory
cargo run --example chatbot

# Terminal chat - interactive chatbot with persistent memory (TRY THIS!)
cargo run --example terminal_chat

# Persistence - save/load memory state (binary, JSON, compressed)
cargo run --example persistence

# Rich data - emotional states, user facts, preferences
cargo run --example rich_data

# GPU & LLM - GPU acceleration and prompt generation
cargo run --example gpu_and_llm

# Memory server - REST API for memory operations
cargo run --example memory_server --release

# Benchmark - performance testing with/without parallel processing
cargo run --example benchmark --release

# Run tests
cargo test

# Run with release optimizations
cargo run --release
```

## Docker Deployment

Run the API server in Docker with persistent storage:

```bash
# Start the server
docker-compose up -d

# Check status
curl http://localhost:3000/health

# Create a memory
curl -X POST http://localhost:3000/api/v1/memories \
  -H 'Content-Type: application/json' \
  -d '{"width": 100, "height": 100}'

# Stop the server
docker-compose down
```

All memory files are persisted in `./data/` directory.

📖 **Full Docker guide:** [DOCKER.md](DOCKER.md)

## Applications

### AI and Neural Networks
- **Spatial memory**: Represent environments without edge artifacts
- **Attention mechanisms**: Spreading activation models
- **Recurrent patterns**: Memory that naturally cycles
- **Emotional intelligence**: Track user sentiment over time
- **Context management**: Spatially organized conversation history

### Chatbots and Conversational AI
- **Topic clustering**: Related topics stored nearby
- **Emotional tracking**: Sentiment analysis integration (RoBERTa)
- **Personalization**: User preferences and facts
- **Memory decay**: Natural forgetting through diffusion
- **LLM prompt generation**: Rich context from memory state
- **REST API**: HTTP/JSON interface for memory operations

### Game AI
- **Procedural generation**: Seamless world generation
- **NPC behavior**: Continuous spatial reasoning
- **Pathfinding**: No edge cases in navigation

### Cellular Automata
- **Conway's Game of Life**: On a toroidal grid
- **Neural CA**: For pattern generation
- **Reaction-diffusion**: Chemical-like simulations

## Future Directions

- [ ] 3D toroidal structures (3-torus)
- [ ] Attention mechanisms over toroidal memory
- [ ] Reinforcement learning in toroidal environments
- [ ] Graph neural networks with toroidal topology
- [ ] Visualization tools (2D/3D rendering)
- [x] **Parallel processing for large grids** (Rayon-based multi-core support)
- [x] Integration with machine learning frameworks (PyTorch, MLX)
- [x] GPU acceleration (CUDA, Metal, Apple Silicon)
- [x] Sentiment analysis integration (RoBERTa-style)
- [x] Chatbot applications with emotional tracking
- [x] State persistence (binary, JSON, compressed formats)

## Documentation

### 🚀 Getting Started
- 🌟 **[Plain English Guide](docs/PLAIN_ENGLISH_GUIDE.md)** - Non-technical explanation for everyone
- 📖 **[Quick Reference](docs/QUICK_REFERENCE.md)** - TL;DR guide to get started fast
- 🎯 **[Terminal Chat Demo](docs/TERMINAL_CHAT.md)** - Interactive chatbot walkthrough

### 🏗️ Core Documentation
- 🧠 **[Concepts Deep Dive](docs/CONCEPTS.md)** - How it works for AI, memory formation, theory
- 📊 **[Project Summary](docs/PROJECT_SUMMARY.md)** - Complete overview of features and capabilities
- ✅ **[Implementation Status](docs/IMPLEMENTATION_STATUS.md)** - Production readiness verification

### 💻 Application Guides
- 💬 **[Chatbot Application](docs/CHATBOT_APPLICATION.md)** - Conversational AI with toroidal memory
- 💾 **[State Management](docs/STATE_MANAGEMENT.md)** - Persistence, serialization, save/load
- ⚡ **[GPU & LLM Integration](docs/GPU_LLM_INTEGRATION.md)** - GPU acceleration and prompt generation
- 🌐 **[API Documentation](docs/API_DOCUMENTATION.md)** - Complete REST API reference (15 endpoints)
- 🚀 **[Parallel Processing](docs/PARALLEL_PROCESSING.md)** - Multi-core CPU optimization (2-8x speedup)

### ⚙️ Operations & Deployment
- 🐳 **[Docker Guide](DOCKER.md)** - Container deployment with docker-compose
- 🧪 **[Docker Test Results](docs/DOCKER_TEST_RESULTS.md)** - Deployment verification and testing
- 🔧 **[Parallel Processing](docs/PARALLEL_PROCESSING.md)** - Multi-core CPU optimization (2-8x speedup)

### 🔬 Research & Comparisons
- 🔬 **[AI Memory Comparison](docs/AI_MEMORY_COMPARISON.md)** - vs. vector DBs, graphs, transformers, etc.
- 🎨 **[Visual Guide](docs/VISUAL_GUIDE.txt)** - ASCII art visualizations and diagrams

### 💡 Examples
See the `examples/` directory for 9 complete, runnable demonstrations:
- `terminal_chat` - Interactive chatbot with persistent memory
- `memory_server` - Production REST API server
- `benchmark` - Performance testing with parallel processing
- `memory_formation` - Episodic memory simulation
- `chatbot` - Simple conversational AI
- `rich_data` - Emotional states and user preferences
- `gpu_and_llm` - GPU acceleration patterns
- `basic` - Core features demonstration
- `advanced` - Complex operations showcase
- `persistence` - Save/load examples

## Mathematical Background

A 2D torus can be represented as the Cartesian product of two circles:
$$T^2 = S^1 \times S^1$$

For a discrete grid of size $w \times h$, coordinate wrapping is:
$$x' = ((x \mod w) + w) \mod w$$
$$y' = ((y \mod h) + h) \mod h$$

This ensures proper wrapping for negative coordinates.

## License

MIT License - Feel free to explore and experiment!

## Contributing

This is an exploratory project. Ideas and contributions welcome!
