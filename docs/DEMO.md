# Quick Demo Script for Terminal Chat

This script demonstrates the terminal chat application capabilities.

## Session 1: First Conversation

```bash
$ cargo run --example terminal_chat
```

```
╔════════════════════════════════════════════════════════════╗
║        Toroidal Memory Chatbot - Terminal Edition         ║
║                                                            ║
║  Your conversations are stored in spatial memory that     ║
║  persists between sessions. Related topics cluster        ║
║  together, and memories naturally fade over time.         ║
╚════════════════════════════════════════════════════════════╝

🆕 Creating new memory for user: user_001
👋 Hello! I'm your toroidal memory chatbot.
💬 I'll remember our conversations and build context over time.

📖 Commands:
  /help      - Show this help message
  /context   - Show current memory state
  /stats     - Show memory statistics
  /clear     - Clear all memory (cannot be undone!)
  /save      - Manually save memory to disk
  /quit      - Exit (automatically saves)

💭 You: I love learning about artificial intelligence!
🤖 Bot: I'm glad to hear that! Tell me more!

💭 You: Machine learning is fascinating, especially neural networks
🤖 Bot: We've been talking about learning and machine. What else would you like to explore?

💭 You: I'm also interested in Python programming for AI
🤖 Bot: That's a good question! It seems related to machine which we've been discussing. I'm a simple chatbot, but I'm happy to continue our conversation!
💾 Memory saved to chat_memory_user_001.json

💭 You: /context

📊 Current Memory State:

😊 Positive Topics:
  • learning (strength: 0.86, sentiment: 0.75)
  • artificial (strength: 0.85, sentiment: 0.75)

📝 Neutral Topics:
  • intelligence (strength: 1.00)
  • machine (strength: 0.95)
  • fascinating (strength: 0.92)
  • especially (strength: 0.89)
  • neural (strength: 0.88)
  • networks (strength: 0.87)
  • interested (strength: 0.82)
  • python (strength: 0.80)

💭 You: /stats

📈 Memory Statistics:
  User ID: user_001
  Conversation turns: 4
  Topics stored: 10
  Memory size: 40×40 = 1600 cells
  Active cells: 58 (3.6% of memory)

💭 You: /quit

👋 Saving memory and exiting...
💾 Memory saved to chat_memory_user_001.json
Goodbye! 👋
```

## Session 2: Resume Later (Memory Persists!)

```bash
$ cargo run --example terminal_chat
```

```
╔════════════════════════════════════════════════════════════╗
║        Toroidal Memory Chatbot - Terminal Edition         ║
║                                                            ║
║  Your conversations are stored in spatial memory that     ║
║  persists between sessions. Related topics cluster        ║
║  together, and memories naturally fade over time.         ║
╚════════════════════════════════════════════════════════════╝

📂 Memory loaded from chat_memory_user_001.json
👋 Hello! I'm your toroidal memory chatbot.
💬 I'll remember our conversations and build context over time.

💭 You: What do we talk about before?
🤖 Bot: I remember we've discussed: learning, artificial, intelligence, machine, fascinating.

💭 You: /context

📊 Current Memory State:

😊 Positive Topics:
  • learning (strength: 0.73, sentiment: 0.75)
  • artificial (strength: 0.72, sentiment: 0.75)

📝 Neutral Topics:
  • intelligence (strength: 0.85)
  • machine (strength: 0.81)
  • fascinating (strength: 0.78)
  • especially (strength: 0.76)
  • neural (strength: 0.75)
  • networks (strength: 0.74)

💭 You: Yes! I'm building a neural network with Python
🤖 Bot: We've been talking about building and neural and python. What else would you like to explore?
💾 Memory saved to chat_memory_user_001.json

💭 You: /context

📊 Current Memory State:

😊 Positive Topics:
  • learning (strength: 0.70, sentiment: 0.75)
  • artificial (strength: 0.69, sentiment: 0.75)

📝 Neutral Topics:
  • building (strength: 1.00)    ← NEW!
  • neural (strength: 0.98)      ← RE-ACTIVATED!
  • network (strength: 0.95)     ← NEW!
  • python (strength: 0.93)      ← RE-ACTIVATED!
  • intelligence (strength: 0.82)
  • machine (strength: 0.78)
```

**Notice:**
- ✅ Old topics (`learning`, `artificial`) have decayed (0.86 → 0.70)
- ✅ Mentioned topics (`neural`, `python`) re-activated (0.75 → 0.98!)
- ✅ New topics (`building`, `network`) added
- ✅ Memory persisted between sessions

## Testing Emotion Tracking

```
💭 You: I'm really frustrated with debugging this code
🤖 Bot: I understand that can be challenging. We've been talking about really and frustrated. What else would you like to explore?

💭 You: /context

📊 Current Memory State:

😟 Concerns/Challenges:
  • frustrated (strength: 1.00, concern: 0.80)
  • debugging (strength: 0.95, concern: 0.80)

📝 Neutral Topics:
  • really (strength: 0.92)
  • code (strength: 0.90)

💭 You: But I love solving these problems!
🤖 Bot: I'm glad to hear that! We've been talking about solving. What else would you like to explore?

💭 You: /context

📊 Current Memory State:

😊 Positive Topics:
  • love (strength: 1.00, sentiment: 0.75)    ← Positive!
  • solving (strength: 0.95, sentiment: 0.75)

😟 Concerns/Challenges:
  • frustrated (strength: 0.85, concern: 0.80)  ← Faded
  • debugging (strength: 0.81, concern: 0.80)
```

**Emotion tracking works!**
- Negative emotions detected and tracked
- Positive emotions override previous negative context
- Both coexist in memory with different spatial positions

## Memory Visualization

The chatbot stores topics in a 40×40 toroidal grid:

```
Position Map (approximate):
┌────────────────────────────────────────┐
│                                        │
│    "learning" (15, 8)                  │
│     ↓ ↓ ↓ diffusion waves              │
│    "machine" (17, 9)  ← nearby!        │
│    "neural" (16, 10)  ← related!       │
│                                        │
│                  "python" (28, 15)     │
│                   ↓ ↓ ↓                │
│                  "code" (29, 16)       │
│                                        │
│  "frustrated" (5, 35)  ← far away      │
│                                        │
└────────────────────────────────────────┘

Related topics cluster together in space!
Diffusion spreads activation to neighbors!
```

## Commands Demo

```
💭 You: /help

📖 Commands:
  /help      - Show this help message
  /context   - Show current memory state
  /stats     - Show memory statistics
  /clear     - Clear all memory (cannot be undone!)
  /save      - Manually save memory to disk
  /quit      - Exit (automatically saves)

💭 You: /save
💾 Memory saved to chat_memory_user_001.json
✅ Memory saved successfully!

💭 You: /clear

⚠️  Are you sure you want to clear all memory? (yes/no): yes
✅ Memory cleared!
```

## Memory File Contents

`chat_memory_user_001.json`:

```json
{
  "activation": [
    [0.0, 0.0, 0.0, ..., 0.85, ...],
    [0.0, 0.1, 0.2, ..., 0.92, ...],
    ...
  ],
  "emotion": [
    [0.0, 0.0, 0.0, ..., 0.75, ...],
    [0.0, 0.0, 0.0, ..., -0.60, ...],
    ...
  ],
  "confidence": [
    [0.8, 0.8, 0.8, ..., 0.95, ...],
    ...
  ],
  "position_labels": {
    "learning": [15, 8],
    "machine": [17, 9],
    "neural": [16, 10],
    "python": [28, 15],
    "frustrated": [5, 35]
  },
  "label_positions": {
    "(15, 8)": "learning",
    "(17, 9)": "machine",
    "(16, 10)": "neural",
    "(28, 15)": "python",
    "(5, 35)": "frustrated"
  },
  "user_id": "user_001",
  "turn_count": 12,
  "width": 40,
  "height": 40
}
```

## Key Features Demonstrated

1. ✅ **Persistence**: Memory saved and restored
2. ✅ **Spatial Clustering**: Related topics near each other
3. ✅ **Memory Decay**: Old topics fade (0.95 → 0.70 → 0.45...)
4. ✅ **Re-activation**: Mentioning old topic brings it back
5. ✅ **Emotion Tracking**: Positive/negative sentiment
6. ✅ **Context Awareness**: Bot knows conversation history
7. ✅ **Auto-save**: Every 5 turns + on exit

---

**Try it yourself!**

```bash
cargo run --example terminal_chat
```

Have a conversation, quit, run again - **your memory persists!** 🧠✨
