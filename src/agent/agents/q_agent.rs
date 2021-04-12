use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use rand::{Rng, SeedableRng};
use rand_pcg::Mcg128Xsl64;

use crate::{
    agent::{self, DataPoint, Transition},
    enviroment::{Discrete, Encode, Message, Status},
    Enviroment, Handler, Policy,
};

use rusty_nn::{
    misc::{
        error::BuilderError,
        simd::{as_scalar, AsScalarExt},
    },
    network::Network,
    optimizer::Optimizer,
};

type ReplayBuffer = agent::ReplayBuffer<Box<[f32]>, Discrete, Option<Target>>;

pub trait Train<O>: Debug
where
    O: Optimizer,
{
    fn train(&mut self, optimizer: &mut O, data: &mut ReplayBuffer);
}

pub trait QFunc: Debug {
    fn eval(&self, reward: f32, predictions: &[f32]) -> f32;
}

pub trait EpsGreedy: Debug {
    fn eval(&self, t: u32) -> f32;
}

pub use trainer::QTrainer;
mod trainer {
    use super::*;
    use rusty_nn::trainer::Logger;

    #[derive(Debug)]
    pub struct QTrainer {
        batch_size: u32,
        epochs: u32,
        rng: Mcg128Xsl64,
        logger: Box<dyn Logger>,
    }

    impl QTrainer {
        pub fn new(batch_size: u32, epochs: u32, seed: u64, logger: Box<dyn Logger>) -> Self {
            Self {
                batch_size,
                epochs,
                rng: SeedableRng::seed_from_u64(seed),
                logger,
            }
        }
    }

    impl<O> Train<O> for QTrainer
    where
        O: Optimizer,
    {
        fn train(&mut self, optimizer: &mut O, data: &mut ReplayBuffer) {
            let batches = data.len() as u32 / self.batch_size;

            for epoch in 0..self.epochs {
                let mut epoch_loss = 0.;
                for batch in 0..batches {
                    let mut n = 0;
                    let mut batch_loss = 0.;
                    while n < self.batch_size {
                        let idx = self.rng.gen_range(0, data.len());
                        let data = &data[idx];
                        if let Some(Target { idx, target }) = data.data() {
                            n += 1;
                            batch_loss += optimizer.process_partial(data.state(), *idx, *target);
                        }
                    }
                    optimizer.update_model();
                    batch_loss = batch_loss / self.batch_size as f32;
                    self.logger.batch_loss(epoch, batch, batch_loss);
                    epoch_loss += batch_loss;
                }
                self.logger.epoch_loss(epoch, epoch_loss / batches as f32)
            }
        }
    }
}

use self::core::QAgentCore;
mod core {
    use super::*;

    pub struct QAgentCore<E, O>
    where
        O: DerefMut,
        O::Target: Sized,
    {
        optimizer: O,
        net2: O::Target,
        rng: Mcg128Xsl64,

        eps: f32,
        q_target: Box<dyn QFunc>,
        trainer: Box<dyn Train<O>>,

        phantom: PhantomData<*const E>,
    }

    impl<E, O> QAgentCore<E, O>
    where
        O: DerefMut,
        O::Target: Sized + Clone,
    {
        pub(super) fn new(
            optimizer: O,
            seed: u64,
            eps: f32,
            q_target: Box<dyn QFunc>,
            trainer: Box<dyn Train<O>>,
        ) -> Self {
            Self {
                net2: optimizer.clone(),
                optimizer,
                rng: Mcg128Xsl64::seed_from_u64(seed),
                eps,
                q_target,
                trainer,
                phantom: PhantomData,
            }
        }
    }

    impl<E, O> QAgentCore<E, O>
    where
        E: Enviroment<Action = Discrete> + Encode,
        O: Optimizer + DerefMut,
        O::Target: Network + Sized,
    {
        fn select_random(&mut self, env: &E) -> Discrete {
            let actions = env.actions();
            let idx = self.rng.gen_range(0, actions.len());
            actions[idx]
        }

        pub(super) fn train(&mut self, memory: &mut ReplayBuffer) {
            self.trainer.train(&mut self.optimizer, memory);
        }

        pub(super) fn sync(&mut self, memory: &mut ReplayBuffer) {
            self.net2.copy_weights(&self.optimizer);
            for mut episode in memory.episodes_mut().map(|i| i.peekable()) {
                while let Some(node) = episode.next() {
                    if let Some(peek) = episode.peek() {
                        match peek.transition() {
                            Transition::Trans { action, reward } => {
                                let target = self.get_target(*action, *reward, peek.state());
                                *node.data_mut() = Some(target);
                            }
                            Transition::First { .. } => {
                                unreachable!("This should never be a first node");
                            }
                        }
                    }
                }
            }
        }

        pub(super) fn get_target(
            &mut self,
            action: Discrete,
            reward: f32,
            state: &[f32],
        ) -> Target {
            let predictions = self.net2.predict(state).as_scalar();
            let q = self.q_target.eval(reward, predictions);
            Target {
                idx: action.inner() as usize,
                target: q,
            }
        }

        pub(super) fn set_eps(&mut self, eps: f32) {
            self.eps = eps;
        }
    }

    impl<E, O> Policy for QAgentCore<E, O>
    where
        E: Enviroment<Action = Discrete> + Encode,
        O: Optimizer + Debug + DerefMut,
        O::Target: Network + Clone,
    {
        type Env = E;

        fn action(&mut self, env: &Self::Env) -> E::Action {
            if self.rng.gen_bool(self.eps as f64) {
                self.select_random(env)
            } else {
                let predictions = self.optimizer.predict(env.encode()).as_scalar();

                // print!("[");
                // for i in predictions {
                //     print!("{}, ", i)
                // }
                // println!("]");

                let act = predictions
                    .iter()
                    .enumerate()
                    .max_by(|x, y| {
                        x.1.partial_cmp(y.1).expect(&format!(
                            "Encountered a NaN while selecting action. Predictions were: {:?}",
                            predictions
                        ))
                    })
                    .expect("Predictions were empty")
                    .0;
                Discrete::new(act as u32)
            }
        }
    }
}

pub struct QAgent<E, O>
where
    O: DerefMut,
    O::Target: Sized,
{
    core: QAgentCore<E, O>,
    memory: ReplayBuffer,
    eps_planner: Box<dyn EpsGreedy>,
    train_every: u32,
    lag: u32,
    t: u32,

    reward: Option<f32>,
    action: Option<Discrete>,
}

impl<E, O> Policy for QAgent<E, O>
where
    E: Enviroment<Action = Discrete> + Encode,
    O: Optimizer + Debug + DerefMut,
    O::Target: Network + Clone,
{
    type Env = E;

    fn action(&mut self, env: &Self::Env) -> E::Action {
        self.core.action(env)
    }
}

impl<E, O> QAgent<E, O>
where
    O: Optimizer,
    O: DerefMut,
    O::Target: Clone,
{
    pub fn new(
        optimizer: O,
        memory: usize,
        train_every: u32,
        lag: u32,
        trainer: Box<dyn Train<O>>,
        epsilon: Box<dyn EpsGreedy>,
        q_target: Box<dyn QFunc>,
    ) -> Self {
        Self {
            core: QAgentCore::new(optimizer, 0, epsilon.eval(0), q_target, trainer),
            memory: ReplayBuffer::new(memory),
            eps_planner: epsilon,
            train_every,
            lag,
            t: 0,
            reward: None,
            action: None,
        }
    }
}

impl<E, O> QAgent<E, O>
where
    E: Enviroment<Action = Discrete> + Encode,
    O: Optimizer,
    O: DerefMut,
    O::Target: Network + Sized,
{
    fn update(&mut self) {
        if self.memory.is_full() {
            // for i in self.memory.iter() {
            //     println!("{:?}", i.data())
            // }
            self.t += 1;
            if self.t % self.train_every == 0 {
                self.core.train(&mut self.memory);
                if self.t % (self.lag * self.train_every) == 0 {
                    self.core.sync(&mut self.memory);
                }
            }
            self.core.set_eps(self.eps_planner.eval(self.t));
        }
    }
}

impl<'a, E, O> Handler<Message<'a, E>> for QAgent<E, O>
where
    E: Enviroment<Action = Discrete, Status = Status> + Encode,
    O: DerefMut + Optimizer,
    O::Target: Network + Clone,
{
    fn handle(&mut self, message: Message<'a, E>) {
        match message {
            Message::Action(action) => {
                self.action.replace(action);
            }
            Message::InvalidAction(action) => {
                self.action.replace(action);
                // TODO make invalid action reward confiurable
                self.reward = Some(-1.);
            }
            Message::Reward(reward) => {
                *self.reward.get_or_insert(0.) += reward;
            }
            Message::State(state) => match state.status() {
                Status::Start => {
                    if let (Some(reward), Some(action)) = (self.reward.take(), self.action.take()) {
                        self.memory.push_result(state, Target {state: , idx: action.into()}, action, reward)
                    }
                    self.memory
                        .begin_episode(state.encode().to_owned().into_boxed_slice(), None);
                }
                Status::Playing | Status::End => {
                    let action = self.action.take().expect("Received no action");
                    let reward = self.reward.take().expect("Received no reward");
                    let target = self.core.get_target(action, reward, state.encode());
                    let last = self.memory.back_mut().expect("Memory buffer is empty");
                    *last.data_mut() = Some(target);
                    let state = state.encode().to_owned().into_boxed_slice();
                    self.memory.push_result(state, None, action, reward);
                    self.update();
                }
            },
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct Target {
    idx: usize,
    target: f32,
}

#[derive(Debug)]
pub struct QBuilder<O> {
    optimizer: Option<O>,
    eps: Option<Box<dyn EpsGreedy>>,
    q_target: Option<Box<dyn QFunc>>,
    train_every: Option<u32>,
    memory: Option<usize>,
    lag: Option<u32>,
    trainer: Option<Box<dyn Train<O>>>,
}

impl<O> Default for QBuilder<O> {
    fn default() -> Self {
        Self {
            optimizer: None,
            eps: None,
            q_target: None,
            train_every: None,
            memory: None,
            lag: None,
            trainer: None,
        }
    }
}

impl<O> QBuilder<O>
where
    O: Optimizer + DerefMut,
    O::Target: Network + Clone,
{
    pub fn new() -> Self {
        Default::default()
    }

    pub fn optimizer(mut self, optimizer: O) -> Self {
        self.optimizer.replace(optimizer);
        self
    }
    pub fn eps(mut self, eps: Box<dyn EpsGreedy>) -> Self {
        self.eps.replace(eps);
        self
    }
    pub fn q_target(mut self, q_target: Box<dyn QFunc>) -> Self {
        self.q_target.replace(q_target);
        self
    }
    pub fn train_every(mut self, train_every: u32) -> Self {
        self.train_every.replace(train_every);
        self
    }
    pub fn memory(mut self, memory: usize) -> Self {
        self.memory.replace(memory);
        self
    }
    pub fn lag(mut self, lag: u32) -> Self {
        self.lag.replace(lag);
        self
    }
    pub fn trainer(mut self, trainer: Box<dyn Train<O>>) -> Self {
        self.trainer.replace(trainer);
        self
    }

    pub fn build<E>(self) -> Result<QAgent<E, O>, BuilderError> {
        let optimizer = self.optimizer.ok_or(BuilderError::new("optimizer"))?;
        let memory = self.memory.ok_or(BuilderError::new("memory_len"))?;
        let train_every = self.train_every.ok_or(BuilderError::new("train_every"))?;
        let lag = self.lag.ok_or(BuilderError::new("lag"))?;
        let trainer = self.trainer.ok_or(BuilderError::new("trainer"))?;
        let epsilon = self.eps.ok_or(BuilderError::new("epsilon"))?;
        let q_target = self.q_target.ok_or(BuilderError::new("q_target"))?;

        Ok(QAgent::new(
            optimizer,
            memory,
            train_every,
            lag,
            trainer,
            epsilon,
            q_target,
        ))
    }
}

pub mod bellman {
    use super::QFunc;

    #[derive(Debug, Clone)]
    pub struct Bellman {
        discount: f32,
    }

    impl Bellman {
        pub fn new(discount: f32) -> Self {
            Self { discount }
        }
    }

    impl QFunc for Bellman {
        fn eval(&self, reward: f32, predictions: &[f32]) -> f32 {
            let max = predictions
                .iter()
                .copied()
                .max_by(|x, y| {
                    x.partial_cmp(y).unwrap_or_else(|| {
                        panic!("Encountered a NaN, predictions were: {:?}", predictions)
                    })
                })
                .expect("Predictions are empty");
            reward + self.discount * max
        }
    }

    #[derive(Debug, Clone)]
    pub struct SoftBellman {
        discount: f32,
    }

    impl SoftBellman {
        pub fn new(discount: f32) -> Self {
            Self { discount }
        }
    }

    impl QFunc for SoftBellman {
        fn eval(&self, reward: f32, predictions: &[f32]) -> f32 {
            let entropy = predictions
                .iter()
                .map(|&f| {
                    if f.is_finite() {
                        f.exp()
                    } else {
                        panic!("Encountered a NaN, predictions were: {:?}", predictions)
                    }
                })
                .sum::<f32>()
                .ln();
            reward + self.discount * entropy
        }
    }
}

pub mod epsilon {
    use super::EpsGreedy;

    #[derive(Debug, Clone)]
    pub struct Static {
        epsilon: f32,
    }

    impl Static {
        pub fn new(epsilon: f32) -> Self {
            Self { epsilon }
        }
    }

    impl EpsGreedy for Static {
        fn eval(&self, _t: u32) -> f32 {
            self.epsilon
        }
    }

    impl<T> EpsGreedy for T
    where
        T: Fn(u32) -> f32 + std::fmt::Debug,
    {
        fn eval(&self, t: u32) -> f32 {
            self(t)
        }
    }
}
