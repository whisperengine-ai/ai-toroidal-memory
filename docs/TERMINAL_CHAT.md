# Terminal Chat with Toroidal Memory

> 📚 **Documentation:** [README](../README.md) | [Chatbot Guide](CHATBOT_APPLICATION.md) | [Quick Reference](QUICK_REFERENCE.md) | [State Management](STATE_MANAGEMENT.md) | [Run It Now!](../examples/terminal_chat.rs)

A simple terminal-based chatbot that uses persistent toroidal memory to build conversation context.

## Features

✅ **Persistent Memory**: Conversations saved to disk and restored between sessions  
✅ **Spatial Organization**: Topics stored in 2D toroidal space (related topics cluster)  
✅ **Emotion Tracking**: Simple sentiment analysis (positive/negative/neutral)  
✅ **Memory Decay**: Older topics naturally fade through diffusion  
✅ **Context Awareness**: Bot remembers what you've discussed  
✅ **Auto-save**: Memory saved every 5 turns + on exit  

## How It Works

### 1. Toroidal Spatial Memory
```
Topics are mapped to positions in a 40×40 toroidal grid:
- "python" → position (15, 23)
- "machine learning" → position (17, 24)  [nearby = related!]
- "pizza" → position (5, 38)  [far away = unrelated]
```

### 2. Multi-Layer Storage
Each position stores:
- **Activation**: How recently/strongly discussed (0.0 - 1.0)
- **Emotion**: Sentiment when mentioned (-1.0 to +1.0)
- **Confidence**: How certain we are (increases with repetition)

### 3. Diffusion Dynamics
```
Turn 1: You mention "Python"
  python: ████████ (strong activation)

Turn 5: Still talking about Python
  python: ██████░░ (slightly faded, but active)

Turn 20: Haven't mentioned Python in a while
  python: ░░      (faded, but still there!)

Turn 21: You say "Python" again
  python: ████████ (re-activated!)
```

### 4. Persistence
Memory is saved to `chat_memory_user_001.json`:
```json
{
  "activation": [[0.0, 0.1, ...], ...],
  "emotion": [[0.0, 0.5, ...], ...],
  "position_labels": {
    "python": [15, 23],
    "machine": [17, 24]
  },
  "turn_count": 42
}
```

## Running the App

```bash
cargo run --example terminal_chat
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/context` | Display current memory state (active topics) |
| `/stats` | Show memory statistics |
| `/save` | Manually save memory to disk |
| `/clear` | Clear all memory (cannot be undone!) |
| `/quit` | Exit and save |

## Example Session

```
╔════════════════════════════════════════════════════════════╗
║        Toroidal Memory Chatbot - Terminal Edition         ║
║                                                            ║
║  Your conversations are stored in spatial memory that     ║
║  persists between sessions. Related topics cluster        ║
║  together, and memories naturally fade over time.         ║
╚════════════════════════════════════════════════════════════╝

👋 Hello! I'm your toroidal memory chatbot.
💬 I'll remember our conversations and build context over time.

📖 Commands:
  /help      - Show this help message
  /context   - Show current memory state
  /stats     - Show memory statistics
  /clear     - Clear all memory (cannot be undone!)
  /save      - Manually save memory to disk
  /quit      - Exit (automatically saves)

💭 You: I love learning about AI and machine learning!
🤖 Bot: I'm glad to hear that! Tell me more!
💾 Memory saved to chat_memory_user_001.json

💭 You: Python is great for data science
🤖 Bot: We've been talking about learning and python. What else would you like to explore?

💭 You: /context

📊 Current Memory State:

😊 Positive Topics:
  • learning (strength: 0.85, sentiment: 0.75)
  • machine (strength: 0.82, sentiment: 0.75)

📝 Neutral Topics:
  • python (strength: 1.00)
  • data (strength: 0.95)
  • science (strength: 0.92)

💭 You: What do you remember about me?
🤖 Bot: I remember we've discussed: learning, machine, python, data, science.

💭 You: /stats

📈 Memory Statistics:
  User ID: user_001
  Conversation turns: 4
  Topics stored: 5
  Memory size: 40×40 = 1600 cells
  Active cells: 47 (2.9% of memory)

💭 You: /quit

👋 Saving memory and exiting...
💾 Memory saved to chat_memory_user_001.json
Goodbye! 👋
```

## Next Session

When you run the chatbot again:

```bash
cargo run --example terminal_chat
```

```
📂 Memory loaded from chat_memory_user_001.json
👋 Hello! I'm your toroidal memory chatbot.
💬 I'll remember our conversations and build context over time.

💭 You: Tell me what we discussed last time
🤖 Bot: I remember we've discussed: learning, machine, python, data, science.
```

**The bot remembers!** 🧠

## How Sentiment Analysis Works

Simple keyword-based approach (in production, use RoBERTa):

**Positive keywords**: love, great, awesome, wonderful, excellent, happy, excited...  
**Negative keywords**: hate, terrible, awful, bad, horrible, sad, frustrated...

```
"I love Python!" → +0.75 (positive)
"This is frustrating" → -0.80 (negative)
"Python is a language" → 0.0 (neutral)
```

## Memory File Location

- **Default**: `chat_memory_user_001.json` (in project root)
- **Format**: JSON (human-readable)
- **Size**: ~5-50KB depending on conversation length

## Customization

Edit `terminal_chat.rs` to change:

```rust
// Memory size (default: 40×40)
let mut bot = ToroidalChatbot::new(user_id, 40);

// Diffusion parameters
DiffusionEngine::new(DiffusionConfig {
    diffusion_rate: 0.25,  // How fast activation spreads
    decay_rate: 0.15,      // How fast memories fade
    threshold: 0.01,       // Minimum activation
})

// Auto-save frequency (default: every 5 turns)
if bot.turn_count % 5 == 0 {
    let _ = bot.save_to_file();
}
```

## Implementation Details

### Topic Extraction
Simple word filtering (removes stop words, keeps words >3 chars):
```rust
"I love learning about AI" 
  → ["love", "learning", "about"] 
  → filtered → ["love", "learning"]
```

### Position Hashing
Topics mapped to positions using hash function:
```rust
hash("python") % 40 = 15  (x coordinate)
hash("python") / 40 % 40 = 23  (y coordinate)
→ position (15, 23)
```

### Diffusion Step
Every message triggers diffusion:
```rust
// Activation spreads to neighbors
// Old memories decay
// Creates natural forgetting curve
diffusion_engine.step(&mut activation_memory);
```

## Limitations

⚠️ **Simple Bot**: Not an LLM, responses are template-based  
⚠️ **Keyword Sentiment**: Not as accurate as RoBERTa  
⚠️ **Single User**: Designed for one user (`user_001`)  
⚠️ **No Context Length Limit**: Memory can grow indefinitely  

## Upgrading to Production

To make this production-ready:

1. **Add Real LLM**: Use OpenAI API, Anthropic, or local model
2. **Better Sentiment**: Integrate RoBERTa or BERT
3. **Topic Extraction**: Use NLP (spaCy, NLTK)
4. **Multi-User**: Add user authentication
5. **Database**: Replace JSON with SQLite/PostgreSQL
6. **Compression**: Use binary or compressed format
7. **UI**: Create web interface or GUI

See `docs/GPU_LLM_INTEGRATION.md` for LLM integration examples!

## Related Examples

- `chatbot.rs` - Simpler chatbot without persistence
- `rich_data.rs` - Multi-layer memory demonstration
- `persistence.rs` - Detailed persistence examples
- `gpu_and_llm.rs` - LLM prompt generation

## Learn More

- **[Chatbot Application Guide](../docs/CHATBOT_APPLICATION.md)**: In-depth chatbot concepts
- **[State Management Guide](../docs/STATE_MANAGEMENT.md)**: Persistence strategies
- **[GPU & LLM Integration](../docs/GPU_LLM_INTEGRATION.md)**: Production LLM patterns

---

**Enjoy chatting with your spatially-organized, emotionally-aware, persistent memory chatbot!** 🤖✨
