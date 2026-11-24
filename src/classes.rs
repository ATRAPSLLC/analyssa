//! A disjoint-set forest over small integers.
//!
//! Classes that merge transitively: a phi's result joined with an operand
//! that is itself another phi's operand puts all three in one class, which a
//! pairwise relation cannot say. Each class has one representative, the
//! root its members find.
//!
//! # Example
//!
//! ```rust
//! use analyssa::Classes;
//!
//! let mut classes = Classes::new(4);
//! classes.union(0, 1);
//! classes.union(1, 2);
//! assert_eq!(classes.find(2), classes.find(0));
//! assert_ne!(classes.find(3), classes.find(0));
//! ```

/// A disjoint-set forest over `0..len`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classes {
    /// Parent link per element; a root points at itself.
    parent: Vec<usize>,
}

impl Classes {
    /// A forest of `len` singleton classes.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    /// The representative of `value`'s class: `value` itself where it is
    /// out of range, a class of its own. Shortens the path it walks.
    pub fn find(&mut self, value: usize) -> usize {
        let mut root = value;
        while let Some(&next) = self.parent.get(root) {
            if next == root {
                break;
            }
            root = next;
        }
        let mut cursor = value;
        while let Some(&next) = self.parent.get(cursor) {
            if next == root {
                break;
            }
            if let Some(slot) = self.parent.get_mut(cursor) {
                *slot = root;
            }
            cursor = next;
        }
        root
    }

    /// Puts `a` and `b` in one class, whose representative stays `a`'s, and
    /// says whether they were apart. An element out of range joins nothing.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let (left, right) = (self.find(a), self.find(b));
        if left == right || left >= self.parent.len() {
            return false;
        }
        match self.parent.get_mut(right) {
            Some(slot) => {
                *slot = left;
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Classes;

    /// **Classes merge transitively and keep the first class's
    /// representative.** Three elements joined pairwise are one class whose
    /// representative is the one the first union kept; an element joined to
    /// nothing is its own; joining two of one class again, or one out of
    /// range, changes nothing.
    #[test]
    fn classes_merge_transitively_and_keep_the_first_representative() {
        let mut classes = Classes::new(5);
        assert!(classes.union(1, 2));
        assert!(classes.union(3, 2));
        assert_eq!(classes.find(1), classes.find(3));
        assert_eq!(classes.find(2), 3, "the first class's root, `3`'s, stays");
        assert_eq!(classes.find(4), 4);
        assert!(!classes.union(1, 3), "already one class");
        assert!(!classes.union(9, 1), "out of range");
        assert!(!classes.union(1, 9), "out of range");
        assert_eq!(classes.find(9), 9);
    }
}
