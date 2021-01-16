pub mod mem_buffer;

use crate::enviroment::Enviroment;

pub trait DataCollector {
    type Env: Enviroment;

    fn begin_episode(&mut self, env: &Self::Env);

    fn push_result(
        &mut self,
        env: &Self::Env,
        action: <Self::Env as Enviroment>::Action,
        reward: f32,
    );
}

#[derive(Clone, Debug)]
pub struct DataPoint<S, A, D> {
    state: S,
    transition: Transition<A>,
    data: D,
}

impl<S, A, D> DataPoint<S, A, D> {
    pub fn state(&self) -> &S {
        &self.state
    }

    pub fn transition(&self) -> &Transition<A> {
        &self.transition
    }

    pub fn data(&self) -> &D {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }
}

#[derive(Clone, Debug)]
pub enum Transition<A> {
    First { len: usize },
    Trans { action: A, reward: f32 },
}

// pub struct Episode<'a, I, S, A>
// where
//     S: 'a,
//     A: 'a,
//     I: Iterator<Item = &'a DataPoint<S, A>>,
// {
//     iter: &'a mut I,
//     len: Option<usize>,
//     consumed: usize,
// }

// impl<'a, I, S, A> Episode<'a, I, S, A>
// where
//     S: 'a,
//     A: 'a,
//     I: Iterator<Item = &'a DataPoint<S, A>> + 'a,
// {
//     pub fn new(iter: &'a mut I) -> Self {
//         Self {
//             iter,
//             len: None,
//             consumed: 0,
//         }
//     }
// }

// impl<'a, I, S, A> Iterator for Episode<'a, I, S, A>
// where
//     S: 'a,
//     A: 'a,
//     I: Iterator<Item = &'a DataPoint<S, A>> + 'a,
// {
//     type Item = &'a DataPoint<S, A>;

//     fn next(&mut self) -> Option<Self::Item> {
//         match self.len {
//             Some(len) => {
//                 if self.consumed < len {
//                     let data_point = self.iter.next().expect("Iterator returned None");
//                     self.consumed += 1;
//                     match data_point.transition {
//                         Transition::Trans { .. } => Some(data_point),
//                         Transition::First { .. } => panic!("Expected a transition data node"),
//                     }
//                 } else {
//                     None
//                 }
//             }

//             None => {
//                 let data_point = self.iter.next().expect("Iterator returned None");
//                 self.consumed += 1;
//                 match data_point.transition {
//                     Transition::First { len } => {
//                         self.len = Some(len);
//                         Some(data_point)
//                     }
//                     Transition::Trans { .. } => panic!("Failed to locate first data node"),
//                 }
//             }
//         }
//     }
// }

// impl<'a, I, S, A> Drop for Episode<'a, I, S, A>
// where
//     S: 'a,
//     A: 'a,
//     I: Iterator<Item = &'a DataPoint<S, A>> + 'a,
// {
//     fn drop(&mut self) {
//         let len = self.len.unwrap_or_else(|| {
//             match self.iter.next().expect("Iterator returned None").transition {
//                 Transition::First { len } => {
//                     self.consumed += 1;
//                     len
//                 }
//                 Transition::Trans { .. } => panic!("Failed to locate first data node"),
//             }
//         });
//         self.iter.nth(len - self.consumed);
//     }
// }

// pub struct Pairs<'a, I, S, A>
// where
//     S: 'a,
//     A: 'a,
//     I: Iterator<Item = &'a DataPoint<S, A>>,
// {
//     iter: I,
// }
