/// Diffusion and Activation Spreading
/// 
/// This module implements diffusion algorithms for spreading activation
/// across toroidal memory, useful for modeling neural-like propagation.
/// 
/// With the 'parallel' feature enabled (default), large grids use multi-core
/// processing via Rayon for significant performance improvements.

use crate::toroidal_memory::ToroidalMemory;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct DiffusionConfig {
    pub diffusion_rate: f64,
    pub decay_rate: f64,
    pub threshold: f64,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        DiffusionConfig {
            diffusion_rate: 0.25,
            decay_rate: 0.05,
            threshold: 0.01,
        }
    }
}

pub struct DiffusionEngine {
    config: DiffusionConfig,
}

impl DiffusionEngine {
    pub fn new(config: DiffusionConfig) -> Self {
        DiffusionEngine { config }
    }

    pub fn with_defaults() -> Self {
        DiffusionEngine {
            config: DiffusionConfig::default(),
        }
    }

    /// Perform one step of diffusion across the toroidal memory
    /// 
    /// Uses parallel processing for grids larger than 50x50 when the 'parallel' feature is enabled.
    pub fn step(&self, memory: &ToroidalMemory<f64>) -> ToroidalMemory<f64> {
        let (width, height) = memory.dimensions();
        let mut new_memory = memory.clone();

        // Use parallel processing for large grids
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 2500; // 50x50
            if width * height >= PARALLEL_THRESHOLD {
                return self.step_parallel(memory);
            }
        }

        // Sequential implementation for small grids or when parallel feature is disabled
        for y in 0..height {
            for x in 0..width {
                let new_value = self.compute_cell_value(memory, x, y);
                new_memory.set(x as isize, y as isize, new_value);
            }
        }

        new_memory
    }

    /// Compute the new value for a single cell
    #[inline]
    fn compute_cell_value(&self, memory: &ToroidalMemory<f64>, x: usize, y: usize) -> f64 {
        let current = memory.get(x as isize, y as isize).unwrap();
        
        // Get 4-connected neighbors (von Neumann neighborhood)
        let neighbors = [
            memory.get(x as isize - 1, y as isize).unwrap(),
            memory.get(x as isize + 1, y as isize).unwrap(),
            memory.get(x as isize, y as isize - 1).unwrap(),
            memory.get(x as isize, y as isize + 1).unwrap(),
        ];

        // Calculate diffusion from neighbors
        let neighbor_sum: f64 = neighbors.iter().map(|&&v| v).sum();
        let diffusion_in = neighbor_sum * self.config.diffusion_rate / 4.0;

        // Calculate diffusion out
        let diffusion_out = current * self.config.diffusion_rate;

        // Apply decay
        let decay = current * self.config.decay_rate;

        // Calculate new value
        let new_value = current + diffusion_in - diffusion_out - decay;
        new_value.max(0.0).min(1.0)
    }

    /// Parallel implementation of diffusion step
    #[cfg(feature = "parallel")]
    fn step_parallel(&self, memory: &ToroidalMemory<f64>) -> ToroidalMemory<f64> {
        let (width, height) = memory.dimensions();
        let mut new_memory = memory.clone();

        // Process rows in parallel
        let new_data: Vec<f64> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                (0..width)
                    .map(|x| self.compute_cell_value(memory, x, y))
                    .collect::<Vec<f64>>()
            })
            .collect();

        // Copy computed values back
        for (idx, &value) in new_data.iter().enumerate() {
            let y = idx / width;
            let x = idx % width;
            new_memory.set(x as isize, y as isize, value);
        }

        new_memory
    }

    /// Run diffusion for multiple steps
    pub fn run(&self, memory: &mut ToroidalMemory<f64>, steps: usize) {
        for _ in 0..steps {
            let new_memory = self.step(memory);
            *memory = new_memory;
        }
    }

    /// Add activation at a specific point
    pub fn activate(memory: &mut ToroidalMemory<f64>, x: isize, y: isize, strength: f64) {
        if let Some(cell) = memory.get_mut(x, y) {
            *cell = (*cell + strength).min(1.0);
        }
    }

    /// Add activation in a radius around a point
    /// 
    /// Uses parallel processing for large radius values when the 'parallel' feature is enabled.
    pub fn activate_radius(
        memory: &mut ToroidalMemory<f64>,
        x: isize,
        y: isize,
        radius: isize,
        strength: f64,
    ) {
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_RADIUS_THRESHOLD: isize = 20;
            if radius >= PARALLEL_RADIUS_THRESHOLD {
                Self::activate_radius_parallel(memory, x, y, radius, strength);
                return;
            }
        }

        // Sequential implementation
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let distance = ((dx * dx + dy * dy) as f64).sqrt();
                if distance <= radius as f64 {
                    let falloff = 1.0 - (distance / radius as f64);
                    let activation = strength * falloff;
                    if let Some(cell) = memory.get_mut(x + dx, y + dy) {
                        *cell = (*cell + activation).min(1.0);
                    }
                }
            }
        }
    }

    /// Parallel implementation of radius activation
    #[cfg(feature = "parallel")]
    fn activate_radius_parallel(
        memory: &mut ToroidalMemory<f64>,
        x: isize,
        y: isize,
        radius: isize,
        strength: f64,
    ) {
        // Collect updates in parallel
        let updates: Vec<(isize, isize, f64)> = (-radius..=radius)
            .into_par_iter()
            .flat_map(|dy| {
                (-radius..=radius)
                    .filter_map(|dx| {
                        let distance = ((dx * dx + dy * dy) as f64).sqrt();
                        if distance <= radius as f64 {
                            let falloff = 1.0 - (distance / radius as f64);
                            let activation = strength * falloff;
                            Some((x + dx, y + dy, activation))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<(isize, isize, f64)>>()
            })
            .collect();

        // Apply updates sequentially (required for memory safety)
        for (px, py, activation) in updates {
            if let Some(cell) = memory.get_mut(px, py) {
                *cell = (*cell + activation).min(1.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diffusion_step() {
        let mut memory = ToroidalMemory::new(5, 5);
        memory.set(2, 2, 1.0);

        let engine = DiffusionEngine::with_defaults();
        let new_memory = engine.step(&memory);

        // Center should have decreased
        assert!(*new_memory.get(2, 2).unwrap() < 1.0);
        
        // Neighbors should have increased
        assert!(*new_memory.get(1, 2).unwrap() > 0.0);
    }

    #[test]
    fn test_activation() {
        let mut memory = ToroidalMemory::new(5, 5);
        DiffusionEngine::activate(&mut memory, 2, 2, 0.5);
        
        assert_eq!(*memory.get(2, 2).unwrap(), 0.5);
    }

    #[test]
    fn test_radius_activation() {
        let mut memory = ToroidalMemory::new(10, 10);
        DiffusionEngine::activate_radius(&mut memory, 5, 5, 2, 1.0);
        
        // Center should be activated
        assert!(*memory.get(5, 5).unwrap() > 0.0);
        
        // Points within radius should be activated
        assert!(*memory.get(4, 5).unwrap() > 0.0);
    }
}
