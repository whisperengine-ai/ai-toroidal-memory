# Parallel Processing Guide

> 📚 **Documentation:** [README](../README.md) | [Performance Guide](PERFORMANCE_GUIDE.md) | [Implementation Status](IMPLEMENTATION_STATUS.md) | [Examples](EXAMPLES_GUIDE.md)

This document explains the multi-core parallel processing capabilities in the toroidal memory library.

## Overview

The library uses [Rayon](https://github.com/rayon-rs/rayon) to provide **automatic multi-core parallelization** for computationally intensive operations. This can provide **2-8x performance improvements** on large grids, scaling with the number of available CPU cores.

## Feature Flag

Parallel processing is controlled by the `parallel` feature flag:

```toml
# In your Cargo.toml

# Parallel enabled (default)
[dependencies]
ai-toroidal-memory = "0.1"

# Parallel disabled
[dependencies]
ai-toroidal-memory = { version = "0.1", default-features = false }
```

## When Parallelization Activates

The library **automatically** chooses between sequential and parallel execution based on problem size:

### Diffusion Operations

- **Small grids (< 50×50)**: Sequential (lower overhead)
- **Large grids (≥ 50×50)**: Parallel (multi-core)

Example:
```rust
let mut memory = ToroidalMemory::new(100, 100); // Uses parallel
let engine = DiffusionEngine::with_defaults();
engine.run(&mut memory, 10); // Runs on multiple cores
```

### Radius Activation

- **Small radius (< 20)**: Sequential
- **Large radius (≥ 20)**: Parallel

Example:
```rust
let mut memory = ToroidalMemory::new(200, 200);
DiffusionEngine::activate_radius(&mut memory, 100, 100, 30, 1.0); // Uses parallel
```

## Performance Characteristics

### Expected Speedup

| Grid Size | Cores | Sequential | Parallel | Speedup |
|-----------|-------|------------|----------|---------|
| 50×50 | 4 | 2.5ms | 2.5ms | ~1x (overhead) |
| 100×100 | 4 | 10ms | 4ms | ~2.5x |
| 200×200 | 4 | 40ms | 12ms | ~3.3x |
| 500×500 | 8 | 250ms | 40ms | ~6.2x |
| 1000×1000 | 8 | 1000ms | 140ms | ~7.1x |

*Actual performance varies by CPU architecture, memory bandwidth, and workload.*

### Why Sequential for Small Grids?

Parallel processing has overhead:
- Thread pool management
- Work distribution
- Result aggregation

For small datasets (< 2500 cells), this overhead exceeds the benefits. The library automatically uses sequential processing to maintain optimal performance.

## How It Works

### Diffusion Parallelization

The diffusion algorithm is **embarrassingly parallel** because each cell's next state depends only on its current neighbors, not on other cells being computed simultaneously.

```rust
// Parallel implementation (simplified)
fn step_parallel(&self, memory: &ToroidalMemory<f64>) -> ToroidalMemory<f64> {
    let new_data: Vec<f64> = (0..height)
        .into_par_iter()  // Parallel row iterator
        .flat_map(|y| {
            (0..width)
                .map(|x| self.compute_cell_value(memory, x, y))
                .collect::<Vec<f64>>()
        })
        .collect();
    
    // ... copy results back
}
```

**Key points:**
- Read-only access to source memory (thread-safe)
- Independent computation per cell
- Results collected and applied atomically

### Radius Activation Parallelization

```rust
// Parallel implementation (simplified)
fn activate_radius_parallel(...) {
    // Compute updates in parallel
    let updates: Vec<(x, y, value)> = (-radius..=radius)
        .into_par_iter()  // Parallel range iterator
        .flat_map(|dy| { /* compute activations */ })
        .collect();
    
    // Apply updates sequentially (required for memory safety)
    for (x, y, activation) in updates {
        memory.set(x, y, activation);
    }
}
```

## Benchmarking

Run the included benchmark to measure performance on your system:

```bash
# With parallel processing (default)
cargo run --example benchmark --release

# Without parallel processing
cargo run --example benchmark --release --no-default-features
```

Example output:
```
╔════════════════════════════════════════════════════════════╗
║     Toroidal Memory Performance Benchmarks                ║
╚════════════════════════════════════════════════════════════╝

✅ Parallel processing: ENABLED
Available CPU cores: 14

📊 Diffusion Benchmarks (10 steps)
═══════════════════════════════════════════════════════════
Small (10×10)           0.000s  (94491165 cells/sec)
Medium (50×50)          0.003s  (8205578 cells/sec)
Large (100×100)         0.005s  (20271126 cells/sec)
Very Large (200×200)    0.008s  (47366242 cells/sec)
Huge (500×500)          0.024s  (102503126 cells/sec)
```

## Thread Pool Configuration

Rayon automatically detects and uses all available CPU cores. You can control this with environment variables:

```bash
# Use 4 threads
RAYON_NUM_THREADS=4 cargo run --example benchmark --release

# Use 1 thread (effectively sequential)
RAYON_NUM_THREADS=1 cargo run --example benchmark --release
```

## Best Practices

### When to Use Parallel Processing

✅ **Good candidates:**
- Large memory grids (≥ 100×100)
- Many diffusion steps (≥ 10 steps)
- Large radius activations (radius ≥ 20)
- Production systems with multi-core CPUs

❌ **Poor candidates:**
- Small grids (< 50×50)
- Single-threaded environments
- Embedded systems with limited resources
- When deterministic timing is critical

### Optimizing Performance

1. **Batch operations**: Run multiple diffusion steps at once
   ```rust
   engine.run(&mut memory, 100); // Better than 100 separate calls
   ```

2. **Use release builds**: Debug builds are 10-100x slower
   ```bash
   cargo build --release
   ```

3. **Size appropriately**: Use grid sizes that align with your data
   ```rust
   // Good: Leverages parallelism
   let memory = ToroidalMemory::new(100, 100);
   
   // Suboptimal: Too small for parallel benefit
   let memory = ToroidalMemory::new(10, 10);
   ```

4. **Profile your workload**: Use benchmarks to verify improvements
   ```bash
   cargo run --example benchmark --release
   ```

## Disabling Parallel Processing

To disable parallel processing entirely:

```toml
# In Cargo.toml
[dependencies]
ai-toroidal-memory = { version = "0.1", default-features = false }
```

Or build without default features:
```bash
cargo build --no-default-features
```

This removes the Rayon dependency and all parallel code paths, resulting in a smaller binary and simpler code.

## Platform Considerations

### macOS (Apple Silicon)

Apple Silicon (M1/M2/M3) has excellent parallel performance:
- High core count (8-16 cores typical)
- Unified memory architecture (low latency)
- Excellent SIMD support

**Expected speedup**: 4-8x on large grids

### Linux

Performance varies by CPU:
- Server CPUs (≥ 8 cores): 6-8x speedup
- Desktop CPUs (4-6 cores): 3-5x speedup
- Low-power CPUs (2-4 cores): 1.5-3x speedup

### Windows

Similar to Linux, performance scales with core count.

## Memory Considerations

Parallel processing has memory implications:

1. **Thread stacks**: Each thread needs stack space (~2MB default)
2. **Work-stealing**: Rayon uses work-stealing queues (small overhead)
3. **Result buffers**: Temporary vectors for collecting results

For very large grids (>1000×1000), ensure adequate RAM:
- 1000×1000 grid ≈ 8MB per f64 layer
- With temporary buffers: ~16MB during diffusion

## Future Optimizations

Potential future improvements:

1. **SIMD vectorization**: Explicit SIMD for 2-4x additional speedup
2. **GPU acceleration**: CUDA/Metal for 10-100x on suitable hardware
3. **Adaptive thresholds**: Dynamic selection based on CPU characteristics
4. **Lock-free updates**: Atomic operations for certain patterns

## Troubleshooting

### "Parallel is slower than sequential"

This is normal for small grids due to overhead. The threshold (50×50) is conservative. If you have a very fast CPU, parallel may not help until larger grids (100×100+).

### "Using all CPU cores but no speedup"

Check:
1. Are you in release mode? (`--release`)
2. Is your grid large enough? (≥ 50×50)
3. Is memory bandwidth the bottleneck? (Profile with `perf` on Linux)

### "Want to force parallel for testing"

Modify the threshold in `src/diffusion.rs`:
```rust
const PARALLEL_THRESHOLD: usize = 1; // Force parallel for all sizes
```

## Summary

- ✅ **Automatic**: No code changes needed
- ✅ **Safe**: Rust's ownership prevents data races
- ✅ **Fast**: 2-8x speedup on typical multi-core systems
- ✅ **Optional**: Can be disabled with feature flag
- ✅ **Smart**: Only activates when beneficial

The library automatically provides optimal performance across different problem sizes and hardware configurations.

---

**See Also:**
- [Rayon Documentation](https://docs.rs/rayon/)
- [examples/benchmark.rs](../examples/benchmark.rs)
- [README.md](../README.md)
