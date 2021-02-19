use std::{
    collections::vec_deque::{self, VecDeque},
    iter::{ExactSizeIterator, FusedIterator},
    ops::{Deref, DerefMut, Index},
};

#[derive(Clone, Debug)]
pub struct DataPoint<S, A, D> {
    state: S,
    transition: Transition<A>,
    pub data: D,
}

impl<S, A, D> DataPoint<S, A, D> {
    /// Get a reference to the data point's state.
    pub fn state(&self) -> &S {
        &self.state
    }

    /// Get a reference to the data point's transition.
    pub fn transition(&self) -> &Transition<A> {
        &self.transition
    }

    /// Get a reference to the data point's data.
    pub fn data(&self) -> &D {
        &self.data
    }

    /// Get a mutable reference to the data point's data.
    pub fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }
}

#[derive(Clone, Debug)]
pub enum Transition<A> {
    First { len: usize },
    Trans { action: A, reward: f32 },
}

// TODO dont rely on the VecDeque's capacity
#[derive(Debug)]
pub struct ReplayBuffer<S, A, D> {
    buffer: VecDeque<DataPoint<S, A, D>>,
    capacity: usize,
    head: usize,
}

impl<S, A, D> Clone for ReplayBuffer<S, A, D>
where
    S: Clone,
    A: Clone,
    D: Clone,
{
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            capacity: self.capacity,
            head: self.head,
        }
    }
}

impl<S, A, D> ReplayBuffer<S, A, D> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            head: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.capacity
    }

    pub fn episodes(&self) -> Episodes<'_, S, A, D> {
        Episodes {
            buffer: &self.buffer,
            head: 0,
        }
    }

    pub fn episodes_mut(&mut self) -> EpisodesMut<'_, S, A, D> {
        EpisodesMut {
            buffer: &mut self.buffer,
            head: 0,
        }
    }

    pub fn iter(&self) -> vec_deque::Iter<'_, DataPoint<S, A, D>> {
        self.buffer.iter()
    }

    pub fn iter_mut(&mut self) -> vec_deque::IterMut<'_, DataPoint<S, A, D>> {
        self.buffer.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn front(&self) -> Option<&DataPoint<S, A, D>> {
        self.buffer.front()
    }

    pub fn front_mut(&mut self) -> Option<&mut DataPoint<S, A, D>> {
        self.buffer.front_mut()
    }

    pub fn back(&self) -> Option<&DataPoint<S, A, D>> {
        self.buffer.back()
    }

    pub fn back_mut(&mut self) -> Option<&mut DataPoint<S, A, D>> {
        self.buffer.back_mut()
    }

    fn truncate_data(&mut self, old: Option<DataPoint<S, A, D>>) {
        if let Some(data_point) = old {
            match data_point.transition {
                Transition::First { len } => {
                    self.buffer.len();
                    let first = self
                        .buffer
                        .front_mut()
                        .expect("Call me when the buffer isn't empty");
                    if let Transition::Trans { .. } = first.transition {
                        first.transition = Transition::First { len: len - 1 }
                    }
                }
                Transition::Trans { .. } => {
                    panic!("Timeline is broken, could not locate first data node")
                }
            }
        }
    }

    fn push_data(&mut self, data: DataPoint<S, A, D>) {
        if self.is_full() {
            let old = self.buffer.pop_front();
            self.head = self.head.saturating_sub(1);
            self.truncate_data(old);
        }
        self.buffer.push_back(data);
        match &mut self.buffer[self.head].transition {
            Transition::First { len } => *len += 1,
            Transition::Trans { .. } => {
                panic!("Head does not point to first data node")
            }
        }
    }

    pub fn begin_episode(&mut self, state: S, data: D) {
        let data_point = DataPoint {
            state,
            transition: Transition::First { len: 0 },
            data,
        };
        // the index of the first data node will be the current buffer length
        self.head = self.buffer.len();
        self.buffer.capacity();
        self.push_data(data_point);
    }

    pub fn push_result(&mut self, state: S, data: D, action: A, reward: f32) {
        let data_point = DataPoint {
            state,
            transition: Transition::Trans { action, reward },
            data,
        };

        self.push_data(data_point);
    }
}

impl<S, A, D, T> Index<T> for ReplayBuffer<S, A, D>
where
    VecDeque<DataPoint<S, A, D>>: Index<T>,
{
    type Output = <VecDeque<DataPoint<S, A, D>> as Index<T>>::Output;

    fn index(&self, index: T) -> &Self::Output {
        self.buffer.index(index)
    }
}

pub use episodes::{Episode, Episodes};
mod episodes {
    use super::*;
    pub struct Episodes<'a, S, A, D> {
        pub(super) buffer: &'a VecDeque<DataPoint<S, A, D>>,
        pub(super) head: usize,
    }

    impl<'a, S, A, D> Iterator for Episodes<'a, S, A, D> {
        type Item = Episode<'a, S, A, D>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.head < self.buffer.len() {
                let len = match self.buffer[self.head].transition {
                    Transition::First { len } => len,
                    Transition::Trans { .. } => panic!("Could not locate first data node"),
                };
                let head = self.head;
                self.head += len;
                Some(Episode {
                    iter: self.buffer.range(head..self.head).into_iter(),
                })
            } else {
                None
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, Some(self.buffer.len()))
        }
    }

    impl<'a, S, A, D> FusedIterator for Episodes<'a, S, A, D> {}

    pub struct Episode<'a, S, A, D> {
        iter: vec_deque::Iter<'a, DataPoint<S, A, D>>,
    }

    impl<'a, S, A, D> Episode<'a, S, A, D> {
        // pub fn pairs(self) -> Pairs<'a, S, A, D> {
        //     Pairs {
        //         iter: self.iter.peekable(),
        //     }
        // }
    }

    impl<'a, S, A, D> Iterator for Episode<'a, S, A, D> {
        type Item = &'a DataPoint<S, A, D>;

        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next()
        }
    }

    impl<'a, S, A, D> FusedIterator for Episode<'a, S, A, D> where
        vec_deque::Iter<'a, DataPoint<S, A, D>>: FusedIterator
    {
    }

    impl<'a, S, A, D> ExactSizeIterator for Episode<'a, S, A, D>
    where
        vec_deque::Iter<'a, DataPoint<S, A, D>>: ExactSizeIterator,
    {
        fn len(&self) -> usize {
            self.iter.len()
        }
    }

    impl<'a, S, A, D> DoubleEndedIterator for Episode<'a, S, A, D> {
        fn next_back(&mut self) -> Option<Self::Item> {
            self.iter.next_back()
        }
    }
}

pub use episodes_mut::{EpisodeMut, EpisodesMut};
mod episodes_mut {
    use super::*;
    pub struct EpisodesMut<'a, S, A, D> {
        pub(super) buffer: &'a mut VecDeque<DataPoint<S, A, D>>,
        pub(super) head: usize,
    }

    impl<'a, S, A, D> Iterator for EpisodesMut<'a, S, A, D> {
        type Item = EpisodeMut<'a, S, A, D>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.head < self.buffer.len() {
                let len = match self.buffer[self.head].transition {
                    Transition::First { len } => len,
                    Transition::Trans { .. } => panic!("Could not locate first data node"),
                };
                let head = self.head;
                self.head += len;
                Some(EpisodeMut {
                    // TODO pretty sure this isn't correct
                    iter: unsafe {
                        std::mem::transmute(self.buffer.range_mut(head..self.head).into_iter())
                    },
                })
            } else {
                None
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, Some(self.buffer.len()))
        }
    }

    impl<'a, S, A, D> FusedIterator for EpisodesMut<'a, S, A, D> {}

    pub struct EpisodeMut<'a, S, A, D> {
        iter: vec_deque::IterMut<'a, DataPoint<S, A, D>>,
    }

    impl<'a, S, A, D> EpisodeMut<'a, S, A, D> {
        // pub fn pairs(self) -> Pairs<'a, S, A, D> {
        //     Pairs {
        //         iter: self.iter.peekable(),
        //     }
        // }
    }

    impl<'a, S, A, D> Iterator for EpisodeMut<'a, S, A, D> {
        type Item = &'a mut DataPoint<S, A, D>;

        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next()
        }
    }

    impl<'a, S, A, D> FusedIterator for EpisodeMut<'a, S, A, D> where
        vec_deque::Iter<'a, DataPoint<S, A, D>>: FusedIterator
    {
    }

    impl<'a, S, A, D> ExactSizeIterator for EpisodeMut<'a, S, A, D>
    where
        vec_deque::IterMut<'a, DataPoint<S, A, D>>: ExactSizeIterator,
    {
        fn len(&self) -> usize {
            self.iter.len()
        }
    }

    impl<'a, S, A, D> DoubleEndedIterator for EpisodeMut<'a, S, A, D> {
        fn next_back(&mut self) -> Option<Self::Item> {
            self.iter.next_back()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buffer() -> ReplayBuffer<usize, (), u32> {
        let mut buffer = ReplayBuffer::new(6);

        let mut num = 0..;
        for (i, len) in [1, 2, 3, 2].iter().enumerate() {
            buffer.begin_episode(i, num.next().unwrap());
            for _ in 0..len - 1 {
                buffer.push_result(i, num.next().unwrap(), (), 0.);
            }
        }
        buffer
    }

    #[test]
    fn buffer_test() {
        let buffer = make_buffer();
        assert_eq!(buffer.len(), 6);
        for (correct, episode) in [
            vec![(1, 2)],
            vec![(2, 3), (2, 4), (2, 5)],
            vec![(3, 6), (3, 7)],
        ]
        .iter()
        .zip(buffer.episodes())
        {
            let episode = episode.collect::<Vec<_>>();
            assert_eq!(correct.len(), episode.len());
            for ((e, i), DataPoint { state, data, .. }) in correct.iter().zip(episode) {
                assert_eq!((e, i), (state, data));
            }
        }
    }
}
