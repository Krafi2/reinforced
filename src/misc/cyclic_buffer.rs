use std::{collections::VecDeque, ops::{Index, IndexMut, Range}};

pub struct CyclicBuffer<D> {
    pub(crate) buffer: Vec<D>,
    pub(crate) idx: usize,
}

impl<D> CyclicBuffer<D> {
    pub fn new(size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(size),
            idx: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn iter<'a>(&'a self) -> Cycle<'a, D> {
        self.into_iter()
    }

    pub fn slice<'a>(&'a self, range: Range<usize>) -> Slice<'a, D> {
        assert!(range.start <= range.end);
        assert!(range.end <= self.buffer.len());
        let idx = if self.idx < self.buffer.len() {
            self.idx
        } else {
            0
        };
        Slice {
            slice: &self.buffer,
            first: (idx + range.start) % self.buffer.len(),
            len: range.end - range.start,
        }
    }

    pub fn push(&mut self, mut data: D) -> Option<D> {
        let mut discard = None;
        // push if the buffer isn't full yet, otherwise overwrite
        if self.buffer.len() < self.buffer.capacity() {
            self.buffer.push(data);
        } else {
            std::mem::swap(&mut self.buffer[self.idx], &mut data);
            discard = Some(data);
        }
        self.idx += 1;
        // wrap to the beginning of the buffer
        if self.idx >= self.buffer.len() {
            self.idx = 0;
        }
        discard
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.buffer.capacity()
    }

    pub fn first(&self) -> Option<&D> {
        self.buffer.get(self.idx).or_else(|| self.buffer.first())
    }

    pub fn first_mut(&mut self) -> Option<&mut D> {
        let idx = if self.idx < self.len() { self.idx } else { 0 };
        self.buffer.get_mut(idx)
    }

    pub fn last(&self) -> Option<&D> {
        let idx = self.idx.checked_sub(1).unwrap_or_else(|| self.len() - 1);
        self.buffer.get(idx)
    }

    pub fn last_mut(&mut self) -> Option<&mut D> {
        let idx = self.idx.checked_sub(1).unwrap_or_else(|| self.len() - 1);
        self.buffer.get_mut(idx)
    }
}

impl<'a, D> Index<usize> for CyclicBuffer<D> {
    type Output = D;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len());
        &self.buffer[(self.idx + index) % self.len()]
    }
}

impl<'a, D> IndexMut<usize> for CyclicBuffer<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len());
        let idx = (self.idx + index) % self.len();
        &mut self.buffer[idx]
    }
}

impl<'a, D> IntoIterator for &'a CyclicBuffer<D> {
    type Item = &'a D;

    type IntoIter = Cycle<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        Cycle {
            slice: &self.buffer,
            end: self.idx,
            idx: (self.idx + 1) % self.buffer.len(),
            stop_iter: false,
        }
    }
}

// macro_rules! iterator {
//     (struct $name:tt -> $t:ty, $inner:ty, $ref:tt ) => {
//         #[derive(Debug)]
//         pub struct $name<'a, T> {
//             slice: $inner,
//             end: usize,
//             idx: usize,
//             stop_iter: bool,
//         }

//         impl<'a, T> $name<'a, T>
//         where T: AsMut<>
//         {
//             /// Returns a reference to the underlying slice
//             pub fn slice(&'a self) -> &'a [T] {
//                 self.slice
//             }

//             pub fn slice_mut(&'a self) -> &'a mut [T]
//             where
//                 $inner: AsMut<[T]>
//             {
//                 self.slice.as_mut()
//             }

//             /// Returns the index which will be returned last by the iterator
//             pub fn end(&self) -> usize {
//                 self.end
//             }
//         }

//         impl<'a, T> Iterator for $name<'a, T> {
//             type Item = $t;

//             fn next(&mut self) -> Option<Self::Item> {
//                 if self.stop_iter {
//                     return None;
//                 }

//                 if self.idx == self.end {
//                     self.stop_iter = true;
//                 }

//                 let res = Some(self.slice[self.idx].$ref());

//                 self.idx += 1;
//                 if self.idx == self.slice.len() {
//                     self.idx = 0;
//                 }
//                 res
//             }

//             fn size_hint(&self) -> (usize, Option<usize>) {
//                 let len = self.len();
//                 (len, Some(len))
//             }

//             fn nth(&mut self, n: usize) -> Option<Self::Item> {
//                 if self.stop_iter {
//                     return None;
//                 }

//                 let new_idx = (self.idx + n) % self.slice.len();
//                 if self.idx <= self.end && new_idx >= self.end {
//                     self.stop_iter = true;
//                     // we've overshot the end of the buffer
//                     if new_idx > self.end {
//                         return None;
//                     }
//                 }
//                 self.idx = new_idx;
//                 Some(self.slice[new_idx].$ref())
//             }
//         }

//         impl<'a, T> std::iter::FusedIterator for $name<'a, T> {}

//         impl<'a, T> ExactSizeIterator for $name<'a, T> {
//             fn len(&self) -> usize {
//                 (self.idx + self.slice.len() - self.end - 1) % self.slice.len() + 1
//             }
//         }
//     };
//     (expand $($t:tt)*) => {
//         $($t)*
//     }
// }

// iterator!(struct Cycle -> &'a T, &'a [T], as_ref);
// iterator!(struct CycleMut -> &'a mut T, &'a mut [T], as_mut);

#[derive(Clone, Debug)]
pub struct Cycle<'a, D> {
    slice: &'a [D],
    end: usize,
    idx: usize,
    stop_iter: bool,
}

impl<'a, D> Cycle<'a, D> {
    /// Returns a reference to the underlying slice
    pub fn slice(&self) -> &'a [D] {
        self.slice
    }

    /// Returns the index which will be returned last by the iterator
    pub fn end(&self) -> usize {
        self.end
    }

    pub fn peek(&self) -> Option<<Self as Iterator>::Item> {
        if self.stop_iter {
            None
        } else {
            Some(&self.slice[self.idx])
        }
    }
}

impl<'a, D> Iterator for Cycle<'a, D> {
    type Item = &'a D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stop_iter {
            return None;
        }

        if self.idx == self.end {
            self.stop_iter = true;
        }

        let res = Some(&self.slice[self.idx]);

        self.idx += 1;
        if self.idx == self.slice.len() {
            self.idx = 0;
        }
        res
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.stop_iter {
            return None;
        }

        let new_idx = (self.idx + n) % self.slice.len();
        if self.idx <= self.end && new_idx >= self.end {
            self.stop_iter = true;
            // we've overshot the end of the buffer
            if new_idx > self.end {
                return None;
            }
        }
        self.idx = new_idx;
        Some(&self.slice[new_idx])
    }
}

impl<'a, D> std::iter::FusedIterator for Cycle<'a, D> {}

impl<'a, D> ExactSizeIterator for Cycle<'a, D> {
    fn len(&self) -> usize {
        (self.idx + self.slice.len() - self.end - 1) % self.slice.len() + 1
    }
}

#[derive(Clone, Debug)]
pub struct Slice<'a, D> {
    slice: &'a [D],
    first: usize,
    len: usize,
}

impl<'a, D> Slice<'a, D> {
    pub fn new(slice: &'a [D], first: usize, len: usize) -> Self {
        assert!(first < slice.len());
        assert!(len <= slice.len());

        Self { slice, first, len }
    }
}

impl<'a, D> Index<usize> for Slice<'a, D> {
    type Output = D;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len, "Index out of bounds");
        let idx = (self.first + index) % self.slice.len();
        &self.slice[idx]
    }
}

impl<'a, D> IntoIterator for &Slice<'a, D> {
    type Item = &'a D;

    type IntoIter = Cycle<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        let mut end = (self.first + self.len) % self.slice.len();
        if end < 1 {
            end += self.slice.len() - 1;
        }

        Cycle {
            slice: self.slice,
            end,
            idx: self.first,
            stop_iter: false,
        }
    }
}

// pub struct SliceIter<'a, D> {
//     inner: Slice<'a, D>,
//     idx: usize,
// }

// impl<'a, D> SliceIter<'a, D> {
//     pub fn peek(&self) -> Option<<Self as Iterator>::Item> {
//         if self.idx < self.inner.len {
//             Some(&self.inner.slice[self.idx])
//         } else {
//             None
//         }
//     }
// }

// impl<'a, D> Iterator for SliceIter<'a, D> {
//     type Item = &'a D;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.idx < self.inner.len {
//             let res = &self.inner.slice[self.idx];
//             self.idx += 1;
//             Some(res)
//         } else {
//             None
//         }
//     }

//     fn nth(&mut self, n: usize) -> Option<Self::Item> {
//         self.idx += n;
//         self.next()
//     }

//     fn size_hint(&self) -> (usize, Option<usize>) {
//         let len = self.len();
//         (len, Some(len))
//     }
// }

// impl<'a, D> ExactSizeIterator for SliceIter<'a, D> {
//     fn len(&self) -> usize {
//         self.inner.len.saturating_sub(self.idx)
//     }
// }

// impl<'a, D> FusedIterator for SliceIter<'a, D> {}

// #[derive(Debug)]
// pub struct CycleMut<'a, D> {
//     slice: &'a mut [D],
//     end: usize,
//     idx: usize,
//     stop_iter: bool,
// }

// impl<'a, D> CycleMut<'a, D> {
//     /// Returns a reference to the underlying slice
//     pub fn slice(&'a self) -> &'a [D] {
//         &*self.slice
//     }

//     /// Returns the index which will be returned last by the iterator
//     pub fn end(&self) -> usize {
//         self.end
//     }

//     pub fn peek(&self) -> Option<&D> {
//         if self.stop_iter {
//             None
//         }
//         else {
//             Some(&self.slice[self.idx])
//         }
//     }
// }

// impl<'a, D> Iterator for CycleMut<'a, D> {
//     type Item = &'a mut D;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.stop_iter {
//             return None;
//         }

//         if self.idx == self.end {
//             self.stop_iter = true;
//         }

//         let res = Some(&mut self.slice[self.idx]);

//         self.idx += 1;
//         if self.idx == self.slice.len() {
//             self.idx = 0;
//         }
//         res
//     }

//     fn size_hint(&self) -> (usize, Option<usize>) {
//         let len = self.len();
//         (len, Some(len))
//     }

//     fn nth(&mut self, n: usize) -> Option<Self::Item> {
//         if self.stop_iter {
//             return None;
//         }

//         let new_idx = (self.idx + n) % self.slice.len();
//         if self.idx <= self.end && new_idx >= self.end {
//             self.stop_iter = true;
//             // we've overshot the end of the buffer
//             if new_idx > self.end {
//                 return None;
//             }
//         }
//         self.idx = new_idx;
//         Some(&mut self.slice[new_idx])
//     }
// }

// impl<'a, D> std::iter::FusedIterator for CycleMut<'a, D> {}

// impl<'a, D> ExactSizeIterator for CycleMut<'a, D> {
//     fn len(&self) -> usize {
//         (self.idx + self.slice.len() - self.end - 1) % self.slice.len() + 1
//     }
// }
