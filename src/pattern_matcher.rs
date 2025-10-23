/// Pattern Matching for Toroidal Memory
/// 
/// This module provides pattern recognition and matching capabilities
/// for toroidal memory structures.

use crate::toroidal_memory::ToroidalMemory;

#[derive(Debug, Clone)]
pub struct Pattern<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T: Clone + PartialEq> Pattern<T> {
    pub fn new(width: usize, height: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), width * height, "Data size must match dimensions");
        Pattern { width, height, data }
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        if x < self.width && y < self.height {
            self.data.get(y * self.width + x)
        } else {
            None
        }
    }
}

pub struct PatternMatcher;

impl PatternMatcher {
    /// Find all occurrences of a pattern in toroidal memory
    pub fn find_pattern<T: Clone + Default + PartialEq>(
        memory: &ToroidalMemory<T>,
        pattern: &Pattern<T>,
    ) -> Vec<(isize, isize)> {
        let mut matches = Vec::new();
        let (mem_width, mem_height) = memory.dimensions();
        let (_pat_width, _pat_height) = pattern.dimensions();

        for y in 0..mem_height {
            for x in 0..mem_width {
                if Self::matches_at(memory, pattern, x as isize, y as isize) {
                    matches.push((x as isize, y as isize));
                }
            }
        }

        matches
    }

    /// Check if pattern matches at specific position
    fn matches_at<T: Clone + Default + PartialEq>(
        memory: &ToroidalMemory<T>,
        pattern: &Pattern<T>,
        start_x: isize,
        start_y: isize,
    ) -> bool {
        let (pat_width, pat_height) = pattern.dimensions();

        for py in 0..pat_height {
            for px in 0..pat_width {
                let mem_val = memory.get(start_x + px as isize, start_y + py as isize);
                let pat_val = pattern.get(px, py);

                match (mem_val, pat_val) {
                    (Some(m), Some(p)) if m == p => continue,
                    _ => return false,
                }
            }
        }

        true
    }

    /// Calculate similarity score between two regions (0.0 to 1.0)
    pub fn similarity_score<T: Clone + Default + PartialEq>(
        memory: &ToroidalMemory<T>,
        pattern: &Pattern<T>,
        start_x: isize,
        start_y: isize,
    ) -> f64 {
        let (pat_width, pat_height) = pattern.dimensions();
        let mut matches = 0;
        let mut total = 0;

        for py in 0..pat_height {
            for px in 0..pat_width {
                let mem_val = memory.get(start_x + px as isize, start_y + py as isize);
                let pat_val = pattern.get(px, py);

                total += 1;
                if let (Some(m), Some(p)) = (mem_val, pat_val) {
                    if m == p {
                        matches += 1;
                    }
                }
            }
        }

        if total > 0 {
            matches as f64 / total as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_creation() {
        let pattern = Pattern::new(2, 2, vec![1, 2, 3, 4]);
        assert_eq!(pattern.dimensions(), (2, 2));
        assert_eq!(*pattern.get(0, 0).unwrap(), 1);
        assert_eq!(*pattern.get(1, 1).unwrap(), 4);
    }

    #[test]
    fn test_pattern_matching() {
        let mut memory = ToroidalMemory::new(5, 5);
        memory.set(1, 1, 1);
        memory.set(2, 1, 2);
        memory.set(1, 2, 3);
        memory.set(2, 2, 4);

        let pattern = Pattern::new(2, 2, vec![1, 2, 3, 4]);
        let matches = PatternMatcher::find_pattern(&memory, &pattern);

        assert!(matches.contains(&(1, 1)));
    }

    #[test]
    fn test_similarity_score() {
        let mut memory = ToroidalMemory::new(5, 5);
        memory.set(0, 0, 1);
        memory.set(1, 0, 2);

        let pattern = Pattern::new(2, 1, vec![1, 2]);
        let score = PatternMatcher::similarity_score(&memory, &pattern, 0, 0);

        assert_eq!(score, 1.0);
    }
}
