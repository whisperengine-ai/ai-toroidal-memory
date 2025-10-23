pub mod toroidal_memory;
pub mod pattern_matcher;
pub mod diffusion;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use toroidal_memory::ToroidalMemory;
pub use pattern_matcher::{Pattern, PatternMatcher};
pub use diffusion::{DiffusionEngine, DiffusionConfig};

#[cfg(feature = "gpu")]
pub use gpu::GpuDiffusionEngine;
