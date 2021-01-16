use std::ops::{Deref, DerefMut};

use random_fast_rng::{FastRng, Random};

use crate::enviroment::{
    discrete::{ActionToken, TaggedDiscrete},
    Enviroment, GetToken,
};
use crate::manager::data_collector::{mem_buffer::MemBuffer, DataCollector, DataPoint, Transition};
use crate::{
    agent::{Agent, AgentBuilder},
    enviroment::discrete,
};

use rusty_nn::helpers::AsScalarExt;
use rusty_nn::network::Network;
use rusty_nn::optimizer::Optimizer;
use rusty_nn::trainer::{Processor, Stochaistic, GenericProcessor, TwoVecs, Config};

use std::marker::PhantomData;

pub struct QAgent<E, O, T, D>
where
    E: Enviroment<Action = TaggedDiscrete>,
    O: Optimizer,
    O: DerefMut,
    O::Target: Network + Clone,
    T: Fn(&E) -> D,
{
    optimizer: O,
    net2: O::Target,
    token: ActionToken,

    eps: Box<dyn FnMut(usize) -> f32>,
    q_target: Box<dyn FnMut(f32, &[f32]) -> f32>,

    t: usize,
    age: usize,

    train_every: usize,
    lag: usize,
    config: Config,

    rng: FastRng,

    phantom: PhantomData<*const (E, T, D)>,
}

impl<E, O, T, S> QAgent<E, O, T, S>
where
    E: Enviroment<Action = TaggedDiscrete>,
    O: Optimizer,
    O: DerefMut,
    O::Target: Network + Clone,

    T: Fn(&E) -> S,
    S: AsRef<[f32]>,
{
    fn new<P, Q>(
        optimizer: O,
        train_every: usize,
        lag: usize,
        config: Config,
        token: ActionToken,
        eps: P,
        q: Q,
    ) -> Self
    where
        P: FnMut(usize) -> f32 + 'static,
        Q: FnMut(f32, &[f32]) -> f32 + 'static,
    {
        Self {
            net2: optimizer.clone(),
            optimizer,
            token,
            eps: Box::new(eps),
            q_target: Box::new(q),
            t: 0,
            age: 0,
            train_every,
            lag,
            config,
            rng: FastRng::seed(0, 0),
            phantom: PhantomData,
        }
    }

    fn select_random(&self, env: &E) -> E::Action {
        todo!()
    }

    fn train(&mut self, data: &<Self as Agent>::Data) {
        let processor = Process {
            buffer: data,
            optimizer: &mut self.optimizer,
        };

        let trainer = Stochaistic::new(self.config.batch_size, self.config.epochs, processor);
        trainer.last();

        if data.is_full() && self.t % (self.lag * self.train_every) == 0 {
            self.sync(data)
        }

        // for episode in data.episodes() {
        //     let iter = episode.peekable();
        //     while let Some(state) = iter.next() {
        //         if let Some(&DataPoint { state, data, .. }) = iter.peek() {
                    
        //         }
        //     }
        // }
        // self.optimizer.process_partial(data, index)
    }

    fn sync(&mut self, data: &<Self as Agent>::Data) {
        todo!()
    }
}

struct Process<'a, T, E, S, O> 
where
    T: Fn(&E) -> S,
    E: Enviroment,
{
    buffer: &'a MemBuffer<T, E, S, Option<Target>>,
    optimizer: &'a mut O,
}

impl<'a, T, E, S, O> Processor for Process<'a, T, E, S, O> 
where
    T: Fn(&E) -> S,
    E: Enviroment,
    O: Optimizer,
    S: AsRef<[f32]>,
{
    fn process(&mut self, idx: usize) -> f32 {
        if let DataPoint {state, data: Some(Target { idx, target}), .. } = &self.buffer[idx] {
            self.optimizer.process_partial(state.as_ref(), *idx, *target)
        }
        else {
            0.
        }
    }

    fn size(&self) -> usize {
        self.buffer.len()
    }

    fn end_batch(&mut self, _batch: usize) {
        self.optimizer.update_model();
    }
}

#[derive(Default)]
pub struct Target {
    idx: usize,
    target: f32,
}

impl<E, O, T, S> Agent for QAgent<E, O, T, S>
where
    E: Enviroment<Action = TaggedDiscrete>,
    O: Optimizer,
    O: DerefMut,
    O::Target: Network + Clone,
    T: Fn(&E) -> S,
    S: AsRef<[f32]>,
{
    type Env = E;
    type Data = MemBuffer<T, E, S, Option<Target>>;

    fn action(&mut self, env: &Self::Env, data: &Self::Data) -> <Self::Env as Enviroment>::Action {
        let eps = (self.eps)(self.t);
        let rand = self.rng.get_u32();

        if rand > (eps * u32::MAX as f32) as u32 {
            let rewards = self.optimizer.predict(data.func()(env).as_ref());

            let mut act = 0;
            let mut max = f32::MIN;
            for (i, f) in rewards.as_scalar().iter().enumerate() {
                if *f > max {
                    max = *f;
                    act = i;
                }
            }
            self.token
                .action(act as u32)
                .expect("Could not create action")
        } else {
            self.select_random(env)
        }
    }

    fn update(&mut self, data: &Self::Data) {
        if data.is_full() {
            if self.t % self.train_every == 0 {
                self.train(data);
                self.age += 1;
                if self.age % self.lag == 0 {
                    self.sync(data);
                }
            }
            self.t += 1;
        }
    }
}

pub struct QBuilder<E, O, T, D>
where
    E: Enviroment<Action = TaggedDiscrete>,
    O: Optimizer,
    O: DerefMut,
    O::Target: Network + Clone,
    T: Fn(&E) -> D,
    D: AsRef<[f32]>,
{
    optimizer: Option<O>,
    eps: Option<Box<dyn FnMut(usize) -> f32>>,
    q_target: Option<Box<dyn FnMut(f32, &[f32]) -> f32>>,
    train_every: Option<usize>,
    lag: Option<usize>,
    config: Option<Config>,
    len: Option<usize>,
    func: Option<T>,
    phantom: PhantomData<*const (E, T, D)>,
}

impl<E, O, T, D> QBuilder<E, O, T, D>
where
    E: Enviroment<Action = TaggedDiscrete>,
    O: Optimizer,
    O: DerefMut,
    O::Target: Network + Clone,
    T: Fn(&E) -> D,
    D: AsRef<[f32]>,
{
    pub fn new() -> Self {
        Self {
            optimizer: None,
            eps: None,
            q_target: None,
            train_every: None,
            lag: None,
            config: None,
            len: None,
            func: None,
            phantom: PhantomData,
        }
    }

    pub fn optimizer(mut self, optimizer: O) -> Self {
        self.optimizer.replace(optimizer);
        self
    }
    pub fn eps(mut self, eps: Box<dyn FnMut(usize) -> f32>) -> Self {
        self.eps.replace(eps);
        self
    }
    pub fn q_target(mut self, q_target: Box<dyn FnMut(f32, &[f32]) -> f32>) -> Self {
        self.q_target.replace(q_target);
        self
    }
    pub fn train_every(mut self, train_every: usize) -> Self {
        self.train_every.replace(train_every);
        self
    }
    pub fn lag(mut self, lag: usize) -> Self {
        self.lag.replace(lag);
        self
    }
    pub fn config(mut self, config: Config) -> Self {
        self.config.replace(config);
        self
    }
    pub fn len(mut self, len: usize) -> Self {
        self.len.replace(len);
        self
    }
    pub fn func(mut self, func: T) -> Self {
        self.func.replace(func);
        self
    }
}

impl<E, O, T, S> AgentBuilder<E> for QBuilder<E, O, T, S>
where
    E: Enviroment<Action = TaggedDiscrete>,
    O: Optimizer,
    O: DerefMut,
    O::Target: Network + Clone,
    T: Fn(&E) -> S,
    S: AsRef<[f32]>,
    E: GetToken<Token = ActionToken>,
{
    type Data = MemBuffer<T, E, S, Option<Target>>;
    type Agent = QAgent<E, O, T, S>;

    fn build(self, env: &mut E) -> (Self::Agent, Self::Data) {
        let token = env.get_token();
        let agent = QAgent::new(
            self.optimizer.expect("Value for 'optimizer' not provided"),
            self.train_every
                .expect("Value for 'train_every' not provided"),
            self.lag.expect("Value for 'lag' not provided"),
            self.config.expect("Value for 'config' not provided"),
            token,
            self.eps.expect("Value for 'eps' not provided"),
            self.q_target.expect("Value for 'q_target' not provided"),
        );
        let data = MemBuffer::new(
            self.len.expect("Value for 'len' not provided"),
            self.func.expect("Value for 'func' not provided"),
        );
        (agent, data)
    }
}
