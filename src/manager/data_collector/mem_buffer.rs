use crate::enviroment::Enviroment;

use super::{DataCollector, DataPoint, Transition};

use std::{marker::PhantomData, ops::{Deref, DerefMut}};
use std::{
    collections::vec_deque::{self, VecDeque},
    iter::{ExactSizeIterator, FusedIterator},
};

pub struct MemBuffer<F, E, S, D>
where
    F: Fn(&E) -> S,
    E: Enviroment,
{
    buffer: VecDeque<DataPoint<S, E::Action, D>>,
    func: F,
    head: usize,
    marker: PhantomData<*const E>,
}

impl<F, E, S, D> MemBuffer<F, E, S, D>
where
    F: Fn(&E) -> S,
    E: Enviroment,
{
    pub fn new(size: usize, func: F) -> Self {
        Self {
            buffer: VecDeque::with_capacity(size),
            func,
            head: 0,
            marker: PhantomData,
        }
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.buffer.capacity()
    }

    pub fn func(&self) -> &F {
        &self.func
    }

    pub fn episodes<'a>(&'a self) -> Episodes<'a, S, E::Action, D> {
        Episodes {
            buffer: &self.buffer,
            head: 0,
        }
    }

    pub fn episodes_mut<'a>(&'a mut self) -> EpisodesMut<'a, S, E::Action, D> {
        EpisodesMut {
            buffer: &mut self.buffer,
            head: 0,
        }
    }

    // pub fn iter<'a>(&'a self) -> vec_deque::Iter<'a, DataPoint<S, E::Action, D>> {
    //     self.buffer.iter()
    // }

    // pub fn iter_mut<'a>(&'a mut self) -> vec_deque::IterMut<'a, DataPoint<S, E::Action, D>> {
    //     self.buffer.iter_mut()
    // }

    // pub fn buffer(&self) -> &VecDeque<DataPoint<S, E::Action, D>> {
    //     &self.buffer
    // }

    // pub fn len(&self) -> usize {
    //     self.buffer.len()
    // }

    fn truncate_data(&mut self, old: Option<DataPoint<S, E::Action, D>>) {
        if let Some(data_point) = old {
            match data_point.transition {
                Transition::First { len } => {
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

    fn push_data(&mut self, data: DataPoint<S, <E as Enviroment>::Action, D>) {
        let old = None;
        if self.buffer.len() == self.buffer.capacity() {
            old = self.buffer.pop_front();
            self.head.saturating_sub(1);
        }
        self.truncate_data(old);
        self.buffer.push_back(data);
        match &mut self.buffer[self.head].transition {
            Transition::First { len } => *len += 1,
            Transition::Trans { action, reward } => {
                panic!("Head does not point to first data node")
            }
        }
    }
}

impl<F, E, S, D> DataCollector for MemBuffer<F, E, S, D>
where
    F: Fn(&E) -> S,
    E: Enviroment,
    D: Default
{
    type Env = E;

    fn begin_episode(&mut self, env: &Self::Env) {
        let data_point = DataPoint {
            state: (self.func)(env),
            transition: Transition::First { len: 0 },
            data: Default::default(),
        };
        // the index of the first data node will be the current buffer length
        self.head = self.buffer.len();
        self.push_data(data_point);
    }

    fn push_result(
        &mut self,
        env: &Self::Env,
        action: <Self::Env as Enviroment>::Action,
        reward: f32,
    ) {
        let data_point = DataPoint {
            state: (self.func)(env),
            transition: Transition::Trans { action, reward },
            data: Default::default(),
        };

        self.push_data(data_point);
    }
}

impl<F, E, S, D> Deref for MemBuffer<F, E, S, D>
where
    F: Fn(&E) -> S,
    E: Enviroment,
{
    type Target = VecDeque<DataPoint<S, E::Action, D>>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<F, E, S, D> DerefMut for MemBuffer<F, E, S, D>
where
    F: Fn(&E) -> S,
    E: Enviroment,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

pub use episodes::{Episode, Episodes};
mod episodes {
    use super::*;
    pub struct Episodes<'a, S, A, D> {
        buffer: &'a VecDeque<DataPoint<S, A, D>>,
        head: usize,
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
    
    impl<'a, S, A, D> FusedIterator for Episode<'a, S, A, D> where vec_deque::Iter<'a, DataPoint<S, A, D>>: FusedIterator {}
    
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
        buffer: &'a mut VecDeque<DataPoint<S, A, D>>,
        head: usize,
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
                    iter: unsafe { std::mem::transmute(self.buffer.range_mut(head..self.head).into_iter()) },
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
    
    impl<'a, S, A, D> FusedIterator for EpisodeMut<'a, S, A, D> where vec_deque::Iter<'a, DataPoint<S, A, D>>: FusedIterator {}
    
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

// pub struct Pairs<'a, S, A, D> {
//     iter: Peekable<vec_deque::Iter<'a, DataPoint<S, A, D>>>,
// }

// impl<'a, S, A, D> Iterator for Pairs<'a, S, A, D> {
//     type Item = Pair<'a, S, A, D>;

//     fn next(&mut self) -> Option<Self::Item> {
//         match self.iter.next() {
//             Some(d1) => match self.iter.peek() {
//                 Some(d2) => Some(
//                     Pair::from_data_points(d1, d2)
//                         .expect("Expected a transition node but found a beginning node"),
//                 ),
//                 None => None,
//             },
//             None => None,
//         }
//     }

//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }

// impl<'a, S, A, D> FusedIterator for Pairs<'a, S, A, D> where Cycle<'a, DataPoint<S, A, D>>: FusedIterator {}

// impl<'a, S, A, D> ExactSizeIterator for Pairs<'a, S, A, D>
// where
//     Cycle<'a, DataPoint<S, A, D>>: ExactSizeIterator,
// {
//     fn len(&self) -> usize {
//         self.iter.len()
//     }
// }

// pub struct Pair<'a, S, A, D> {
//     pub state: &'a D,
//     pub next: &'a D,
//     pub action: &'a A,
//     pub reward: f32,
// }

// impl<'a, S, A, D> Pair<'a, S, A, D> {
//     pub fn from_data_points(d1: &'a DataPoint<S, A, D>, d2: &'a DataPoint<S, A, D>) -> Option<Self> {
//         match &d2.transition {
//             Transition::First { .. } => None,
//             Transition::Trans { action, reward } => Some(Pair {
//                 state: &d1.state,
//                 next: &d2.state,
//                 action,
//                 reward: *reward,
//             }),
//         }
//     }
// }
