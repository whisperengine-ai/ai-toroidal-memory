# State Management: How Toroidal Memory is Stored & Retrieved

> 📚 **Documentation:** [README](../README.md) | [API Docs](API_DOCUMENTATION.md) | [Chatbot Guide](CHATBOT_APPLICATION.md) | [Docker](../DOCKER.md) | [Examples](../examples/)

## The Core Question

**Q: How do you save and restore toroidal memory state?**

**A:** Toroidal memory is just a 2D grid of numbers - it can be serialized to disk in multiple formats depending on your needs.

## Memory Layout

Internally, toroidal memory stores data as a flat vector:

```rust
struct ToroidalMemory<T> {
    width: usize,    // e.g., 50
    height: usize,   // e.g., 50
    data: Vec<T>,    // size = width * height = 2,500 elements
}

// Position (x, y) maps to index: y * width + x
```

This makes serialization straightforward!

## Three Storage Formats

### 1. Binary Format (Fastest, Production)

```
File Structure:
┌─────────────┬──────────────┬────────────────────────┐
│ Width (4B)  │ Height (4B)  │ Data (8B per element)  │
└─────────────┴──────────────┴────────────────────────┘

Example for 20x20 grid:
- Header: 8 bytes
- Data: 20 * 20 * 8 = 3,200 bytes
- Total: 3,208 bytes
```

**Pros:**
- ✅ Smallest size (except compressed)
- ✅ Fastest read/write
- ✅ Simple implementation

**Cons:**
- ❌ Not human-readable
- ❌ Platform-dependent (endianness)

**Use when:** Production systems, high-performance needs

### 2. JSON Format (Human-Readable, Debugging)

```json
{
  "width": 20,
  "height": 20,
  "data": [
    0.0, 0.0, 0.1, 0.3, 0.5, ...
  ]
}
```

**Pros:**
- ✅ Human-readable
- ✅ Easy debugging
- ✅ Platform-independent
- ✅ Can edit manually

**Cons:**
- ❌ ~3x larger file size
- ❌ Slower parsing

**Use when:** Development, debugging, sharing with others

### 3. Compressed Format (Smallest, Archival)

```
Binary format + gzip compression

Example compression ratios:
- 50x50 grid (sparse): 20KB → 5KB (74% reduction)
- 50x50 grid (dense):  20KB → 15KB (25% reduction)
```

**Pros:**
- ✅ Smallest file size
- ✅ Good for network transfer
- ✅ Ideal for archival

**Cons:**
- ❌ Slower (compression overhead)
- ❌ More complex

**Use when:** Storing many sessions, network transfer, archival

## Practical Examples

### Example 1: Save/Load Session (Chatbot)

```rust
use ai_toroidal_memory::ToroidalMemory;

// During conversation
let mut memory = ToroidalMemory::new(30, 30);
// ... user interacts, memory gets populated ...

// Save session when user leaves
save_to_binary(&memory, "sessions/user_123.bin")?;

// Restore session when user returns
let memory = load_from_binary("sessions/user_123.bin")?;
// User's conversation context is restored!
```

**File size:** ~7KB for 30x30 grid

### Example 2: Checkpoint System (Training)

```rust
struct CheckpointManager {
    checkpoint_dir: String,
    max_checkpoints: usize,  // Keep only last N
}

// During training
for epoch in 0..100 {
    train_step(&mut memory);
    
    if epoch % 10 == 0 {
        manager.save_checkpoint(&memory, epoch)?;
    }
}

// If training crashes, resume from checkpoint
let (memory, last_epoch) = manager.load_latest_checkpoint()?;
println!("Resuming from epoch {}", last_epoch);
```

**Benefit:** Never lose training progress!

### Example 3: State Snapshots (A/B Testing)

```rust
// Save baseline
let baseline = memory.clone();
save_to_binary(&baseline, "baseline.bin")?;

// Try experiment A
experiment_a(&mut memory);
save_to_binary(&memory, "experiment_a.bin")?;

// Reset to baseline
memory = load_from_binary("baseline.bin")?;

// Try experiment B
experiment_b(&mut memory);
save_to_binary(&memory, "experiment_b.bin")?;

// Compare results
let a_results = analyze("experiment_a.bin");
let b_results = analyze("experiment_b.bin");
```

**Benefit:** Easy experimentation with exact state control

## Performance Characteristics

Based on actual benchmarks (from the example):

| Format | Size (50x50) | Save Speed | Load Speed | Use Case |
|--------|--------------|------------|------------|----------|
| Binary | 20 KB | ⚡⚡⚡ Fast | ⚡⚡⚡ Fast | Production |
| JSON | 58 KB | ⚡⚡ Medium | ⚡⚡ Medium | Development |
| Compressed | 5 KB | ⚡ Slow | ⚡ Slow | Archival |

**Rule of thumb:**
- Binary: Default choice
- JSON: When you need to inspect/debug
- Compressed: When storing 100+ sessions

## Advanced: Incremental State (Delta Encoding)

For very large memories or frequent saves, only store changes:

```rust
struct IncrementalState {
    base_memory: ToroidalMemory<f64>,
    changes: Vec<(isize, isize, f64)>,  // Only changed positions
}

// Instead of saving full 20KB grid every time:
// Save base once: 20 KB
// Save deltas:    ~100 bytes per update

// Example timeline:
// t=0:   Save full state (20 KB)
// t=1:   Save delta: 5 changes (40 bytes)
// t=2:   Save delta: 3 changes (24 bytes)
// ...
// t=100: Save full state (20 KB) ← Periodic full save
```

**Benefit:** 100x-1000x smaller for sparse updates

## State Management Patterns

### Pattern 1: Session-Based (Chatbots)

```
sessions/
  ├── user_001_2025-10-22.session
  ├── user_001_2025-10-23.session
  ├── user_002_2025-10-22.session
  └── user_003_2025-10-22.session

Each file: ~5-10 KB compressed
Retention: Delete after 30 days
```

**Implementation:**
```rust
struct SessionManager {
    sessions_dir: String,
}

impl SessionManager {
    fn save_session(&self, user_id: &str, memory: &Memory) {
        let filename = format!("{}/{}.session", self.sessions_dir, user_id);
        save_compressed(memory, &filename);
    }
    
    fn load_session(&self, user_id: &str) -> Memory {
        let filename = format!("{}/{}.session", self.sessions_dir, user_id);
        load_compressed(&filename)
    }
}
```

### Pattern 2: Checkpoint-Based (Training/Long-Running)

```
checkpoints/
  ├── checkpoint_000000.bin  (oldest - will be deleted)
  ├── checkpoint_000050.bin
  ├── checkpoint_000100.bin  (kept)
  └── checkpoint_000150.bin  (latest)

Keep: Last N checkpoints (e.g., 5)
Auto-cleanup: Delete checkpoints older than N
```

**Implementation:**
```rust
struct CheckpointManager {
    max_checkpoints: usize,
}

impl CheckpointManager {
    fn save_checkpoint(&self, memory: &Memory, step: usize) {
        save_binary(memory, format!("checkpoint_{:06}.bin", step));
        self.cleanup_old_checkpoints();  // Keep only last N
    }
}
```

### Pattern 3: Version Control (Research)

```
experiments/
  ├── baseline_v1.bin
  ├── experiment_attention_v1.bin
  ├── experiment_attention_v2.bin
  ├── experiment_diffusion_v1.bin
  └── best_model.bin

Tag: Which variant performed best
Compare: Load multiple and analyze
```

### Pattern 4: In-Memory Cache + Lazy Persistence

```rust
struct CachedMemory {
    memory: ToroidalMemory<f64>,
    dirty: bool,
    last_save: Instant,
}

impl CachedMemory {
    fn set(&mut self, x: isize, y: isize, value: f64) {
        self.memory.set(x, y, value);
        self.dirty = true;
    }
    
    fn auto_save_if_needed(&mut self) {
        if self.dirty && self.last_save.elapsed() > SAVE_INTERVAL {
            self.save();
            self.dirty = false;
            self.last_save = Instant::now();
        }
    }
}
```

**Benefit:** Performance + durability

## Integration with Databases

### Option 1: File-Based Storage (Simple)

```rust
// Just save to disk
save_to_binary(&memory, "memory_states/session_123.bin")?;

// Pros: Simple, fast
// Cons: No querying, manual management
```

### Option 2: SQLite Blob Storage

```sql
CREATE TABLE memory_states (
    session_id TEXT PRIMARY KEY,
    timestamp INTEGER,
    memory_data BLOB,
    metadata TEXT
);

INSERT INTO memory_states VALUES (
    'user_123',
    1729612800,
    <binary_memory_data>,
    '{"width": 30, "height": 30}'
);
```

**Pros:** Queryable, atomic, transactions
**Cons:** Slightly slower than raw files

### Option 3: Redis (Hot Data)

```rust
// Store in Redis for fast access
redis.set(
    format!("session:{}", user_id),
    serialize(&memory),
    expiration: 24 * 3600  // 24 hours
)?;

// Retrieve
let memory = deserialize(redis.get(format!("session:{}", user_id))?);
```

**Pros:** Extremely fast, distributed
**Cons:** Memory-based (limited by RAM)

### Option 4: Cloud Storage (S3, GCS)

```rust
// Upload to S3
s3_client.put_object(
    bucket: "memory-states",
    key: format!("sessions/{}/state.bin.gz", user_id),
    body: compress(&serialize(&memory))
)?;
```

**Pros:** Scalable, durable, cheap
**Cons:** Network latency

## Best Practices

### 1. Version Your Format

```rust
struct MemorySnapshot {
    version: u32,      // Format version
    width: usize,
    height: usize,
    data: Vec<f64>,
}

// Can migrate old formats
fn load(path: &str) -> Memory {
    match detect_version(path) {
        1 => load_v1(path),
        2 => load_v2(path),
        _ => panic!("Unsupported version"),
    }
}
```

### 2. Include Metadata

```rust
struct MemoryState {
    // Actual memory
    memory: ToroidalMemory<f64>,
    
    // Metadata
    created_at: DateTime,
    last_modified: DateTime,
    user_id: String,
    
    // Diffusion parameters used
    diffusion_config: DiffusionConfig,
}
```

### 3. Validate on Load

```rust
fn load_with_validation(path: &str) -> Result<Memory, Error> {
    let memory = load_from_binary(path)?;
    
    // Sanity checks
    if memory.dimensions().0 == 0 || memory.dimensions().1 == 0 {
        return Err(Error::InvalidDimensions);
    }
    
    // Check for NaN/Inf values
    for y in 0..memory.dimensions().1 {
        for x in 0..memory.dimensions().0 {
            let val = memory.get(x as isize, y as isize).unwrap();
            if !val.is_finite() {
                return Err(Error::InvalidData);
            }
        }
    }
    
    Ok(memory)
}
```

### 4. Automatic Backup

```rust
fn save_with_backup(memory: &Memory, path: &str) -> Result<()> {
    // Backup existing file
    if Path::new(path).exists() {
        std::fs::copy(path, format!("{}.backup", path))?;
    }
    
    // Save new state
    save_to_binary(memory, path)?;
    
    // Remove backup only after successful save
    std::fs::remove_file(format!("{}.backup", path)).ok();
    
    Ok(())
}
```

## Summary

### Storage Formats

| Format | Size | Speed | Use Case |
|--------|------|-------|----------|
| Binary | ⭐ Small | ⚡⚡⚡ Fastest | Production |
| JSON | ⭐⭐ Medium | ⚡⚡ Medium | Debug |
| Compressed | ⭐⭐⭐ Smallest | ⚡ Slowest | Archive |

### Common Patterns

1. **Chatbot Sessions:** Compressed format, auto-save, 30-day retention
2. **Training:** Binary checkpoints, keep last 5, resume on crash
3. **Research:** JSON for inspection, version control, comparisons
4. **Production:** Binary + Redis cache + S3 backup

### Key Takeaways

- ✅ Toroidal memory is just a 2D array - easy to serialize
- ✅ Multiple format options for different use cases
- ✅ Small file sizes (5-20 KB typical)
- ✅ Fast save/load (milliseconds)
- ✅ Enable session persistence, checkpointing, experimentation
- ✅ Combine with databases for production systems

**Try the example:**
```bash
cargo run --example persistence
```
