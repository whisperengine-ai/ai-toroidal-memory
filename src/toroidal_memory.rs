/// Toroidal Memory Structure
/// 
/// This module implements a toroidal (donut-shaped) memory structure where
/// edges wrap around, creating a continuous space without boundaries.
/// Useful for AI pattern recognition and spatial memory modeling.

use std::fmt;

#[derive(Debug, Clone)]
pub struct ToroidalMemory<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T: Clone + Default> ToroidalMemory<T> {
    /// Create a new toroidal memory with given dimensions
    pub fn new(width: usize, height: usize) -> Self {
        ToroidalMemory {
            width,
            height,
            data: vec![T::default(); width * height],
        }
    }

    /// Get the total size of the memory
    pub fn size(&self) -> usize {
        self.width * self.height
    }

    /// Get dimensions (width, height)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Wrap coordinates to handle toroidal topology
    fn wrap_coordinates(&self, x: isize, y: isize) -> (usize, usize) {
        let wrapped_x = ((x % self.width as isize + self.width as isize) % self.width as isize) as usize;
        let wrapped_y = ((y % self.height as isize + self.height as isize) % self.height as isize) as usize;
        (wrapped_x, wrapped_y)
    }

    /// Get value at position with toroidal wrapping
    pub fn get(&self, x: isize, y: isize) -> Option<&T> {
        let (wx, wy) = self.wrap_coordinates(x, y);
        self.data.get(wy * self.width + wx)
    }

    /// Set value at position with toroidal wrapping
    pub fn set(&mut self, x: isize, y: isize, value: T) {
        let (wx, wy) = self.wrap_coordinates(x, y);
        let index = wy * self.width + wx;
        if index < self.data.len() {
            self.data[index] = value;
        }
    }

    /// Get mutable reference at position with toroidal wrapping
    pub fn get_mut(&mut self, x: isize, y: isize) -> Option<&mut T> {
        let (wx, wy) = self.wrap_coordinates(x, y);
        let index = wy * self.width + wx;
        self.data.get_mut(index)
    }

    /// Get neighbors in a given radius (Manhattan distance)
    pub fn get_neighbors(&self, x: isize, y: isize, radius: isize) -> Vec<(isize, isize, &T)> {
        let mut neighbors = Vec::new();
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                let ny = y + dy;
                if let Some(value) = self.get(nx, ny) {
                    neighbors.push((nx, ny, value));
                }
            }
        }
        neighbors
    }

    /// Apply a function to all cells
    pub fn map<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, usize, &mut T),
    {
        for y in 0..self.height {
            for x in 0..self.width {
                let index = y * self.width + x;
                f(x, y, &mut self.data[index]);
            }
        }
    }

    /// Fill the entire memory with a value
    pub fn fill(&mut self, value: T) {
        for item in self.data.iter_mut() {
            *item = value.clone();
        }
    }
}

impl<T: Clone + Default + fmt::Display> fmt::Display for ToroidalMemory<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                let index = y * self.width + x;
                write!(f, "{} ", self.data[index])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let memory: ToroidalMemory<i32> = ToroidalMemory::new(5, 5);
        assert_eq!(memory.size(), 25);
        assert_eq!(memory.dimensions(), (5, 5));
    }

    #[test]
    fn test_wrapping() {
        let mut memory = ToroidalMemory::new(3, 3);
        memory.set(0, 0, 1);
        
        // Test wrapping on negative coordinates
        assert_eq!(*memory.get(-3, -3).unwrap(), 1);
        assert_eq!(*memory.get(-6, -6).unwrap(), 1);
        
        // Test wrapping on positive coordinates
        memory.set(2, 2, 5);
        assert_eq!(*memory.get(5, 5).unwrap(), 5);
    }

    #[test]
    fn test_neighbors() {
        let mut memory = ToroidalMemory::new(5, 5);
        memory.set(2, 2, 10);
        
        let neighbors = memory.get_neighbors(2, 2, 1);
        assert_eq!(neighbors.len(), 8); // 8 neighbors in radius 1
    }
}
