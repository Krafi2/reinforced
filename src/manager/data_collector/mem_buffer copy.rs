use crate::enviroment::Enviroment;
use crate::misc::cyclic_buffer::{Cycle, CyclicBuffer};

use super::{DataCollector, DataPoint, Transition};

use std::iter::{ExactSizeIterator, FusedIterator};
use std::marker::PhantomData;

pub struct MemBuffer<F, E, D>
where
    F: Fn(&E) -> D,
    E: Enviroment,
{
    func: F,
    buffer: CyclicBuffer<DataPoint<D, E::Action>>,
    marker: PhantomData<*const E>,
    first: usize,
}

impl<F, E, D> MemBuffer<F, E, D>
where
    F: Fn(&E) -> D,
    E: Enviroment,
{
    pub fn new(size: usize, func: F) -> Self {
        Self {
            func,
            buffer: CyclicBuffer::new(size),
            marker: PhantomData,
            first: 0,
        }
    }

    pub fn inner(&self) -> &CyclicBuffer<DataPoint<D, E::Action>> {
        &self.buffer
    }

    pub fn is_full(&self) -> bool {
        self.buffer.is_full()
    }

    pub fn func(&self) -> &F {
        &self.func
    }

    pub fn episodes<'a>(&'a self) -> Episodes<'a, D, E::Action> {
        Episodes {
            raw: &self.buffer,
            idx: 0,
        }
    }

    fn truncate_data(&mut self, old: Option<DataPoint<D, E::Action>>) {
        if let Some(data_point) = old {
            match data_point.transition {
                Transition::First { len } => {
                    let first = self
                        .buffer
                        .first_mut()
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
}

impl<F, E, D> DataCollector for MemBuffer<F, E, D>
where
    F: Fn(&E) -> D,
    E: Enviroment,
{
    type Env = E;

    fn begin_episode(&mut self, env: &Self::Env) {
        let data_point = DataPoint {
            state: (self.func)(env),
            transition: Transition::First { len: 1 },
        };
        // the index of the first data node will be buffer.idx, which is the index to be overwritten
        self.first = self.buffer.idx;
        let old = self.buffer.push(data_point);
        self.truncate_data(old);
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
        };
        let old = self.buffer.push(data_point);
        self.truncate_data(old);

        match &mut self.buffer.buffer[self.first].transition {
            Transition::First { len } => *len += 1,
            Transition::Trans { .. } => panic!("Could not locate first data node"),
        };
    }
}

pub struct Episodes<'a, D, A> {
    raw: &'a CyclicBuffer<DataPoint<D, A>>,
    idx: usize,
}

impl<'a, D, A> Episodes<'a, D, A> {
}

impl<'a, D, A> Iterator for Episodes<'a, D, A> {
    type Item = Episode<'a, D, A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.raw.len() {
            let len = match self.raw[self.idx].transition {
                Transition::First { len } => len,
                Transition::Trans { .. } => panic!("Could not locate first data node"),
            };
            self.idx += len;
            Some(Episode {
                iter: self.raw.slice(self.idx..self.idx + len).into_iter(),
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.raw.len()))
    }
}

pub struct Episode<'a, D, A> {
    iter: Cycle<'a, DataPoint<D, A>>,
}

impl<'a, D, A> Episode<'a, D, A> {
    pub fn pairs(self) -> Pairs<'a, D, A> {
        Pairs { iter: self.iter }
    }
}

impl<'a, D, A> Iterator for Episode<'a, D, A> {
    type Item = &'a DataPoint<D, A>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a, D, A> FusedIterator for Episode<'a, D, A> where Cycle<'a, DataPoint<D, A>>: FusedIterator {}

impl<'a, D, A> ExactSizeIterator for Episode<'a, D, A>
where
    Cycle<'a, DataPoint<D, A>>: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, D, A> DoubleEndedIterator for Episode<'a, D, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

pub struct Pairs<'a, D, A> {
    iter: Cycle<'a, DataPoint<D, A>>,
}

impl<'a, D, A> Iterator for Pairs<'a, D, A> {
    type Item = Pair<'a, D, A>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(d1) => match self.iter.peek() {
                Some(d2) => Some(
                    Pair::from_data_points(d1, d2)
                        .expect("Expected a transition node but found a beginning node"),
                ),
                None => None,
            },
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, D, A> FusedIterator for Pairs<'a, D, A> where Cycle<'a, DataPoint<D, A>>: FusedIterator {}

impl<'a, D, A> ExactSizeIterator for Pairs<'a, D, A>
where
    Cycle<'a, DataPoint<D, A>>: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

pub struct Pair<'a, D, A> {
    pub state: &'a D,
    pub next: &'a D,
    pub action: &'a A,
    pub reward: f32,
}

impl<'a, D, A> Pair<'a, D, A> {
    pub fn from_data_points(d1: &'a DataPoint<D, A>, d2: &'a DataPoint<D, A>) -> Option<Self> {
        match &d2.transition {
            Transition::First { .. } => None,
            Transition::Trans { action, reward } => Some(Pair {
                state: &d1.state,
                next: &d2.state,
                action,
                reward: *reward,
            }),
        }
    }
}
