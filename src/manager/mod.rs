pub mod data_collector;

use crate::agent::{Agent, AgentBuilder};
use crate::enviroment::{EnvBuilder, Enviroment, IsTerminal, PlayerRange, SingleWinner};

use data_collector::DataCollector;

pub struct ManagerBuilder<E: EnvBuilder + PlayerRange>
where
    <E::Output as Enviroment>::Action: Clone,
    <E::Output as Enviroment>::Status: IsTerminal,
    E::Output: SingleWinner,
{
    agents: Vec<Box<dyn AgentWrapper<Env = E::Output>>>,
    env: E,
}

impl<E: EnvBuilder + PlayerRange> ManagerBuilder<E>
where
    <E::Output as Enviroment>::Action: Clone,
    <E::Output as Enviroment>::Status: IsTerminal,
    E::Output: SingleWinner,
{
    pub fn new(env: E) -> Self {
        Self {
            agents: Vec::new(),
            env,
        }
    }

    pub fn add_agent<A>(&mut self, agent: A)
    where
        A: AgentBuilder<E>,
        A::Agent: Agent<Env = E::Output> + 'static,
        A::Data: DataCollector<Env = E::Output>,
    {
        let (agent, data_collect) = agent.build(&mut self.env);
        let agent = Box::new(Wrapper::new(agent, data_collect));
        self.agents
            .push(agent as Box<dyn AgentWrapper<Env = <E as EnvBuilder>::Output>>);
    }

    pub fn build(self) -> Option<Manager<E::Output>> {
        let len = self.agents.len();
        if len >= E::MIN && len <= E::MAX.unwrap_or(usize::MAX) {
            Some(Manager {
                agents: self.agents,
                env: self.env.build(),
            })
        } else {
            None
        }
    }
}

pub struct Manager<E>
where
    E: Enviroment,
    E::Status: IsTerminal,
    E: SingleWinner,
{
    agents: Vec<Box<dyn AgentWrapper<Env = E>>>,
    env: E,
}

impl<E> Manager<E>
where
    E: Enviroment,
    E::Status: IsTerminal,
    E: SingleWinner,
{
    pub fn episode(&mut self) -> E::Status {
        self.env.reset();
        loop {
            for (i, agent) in self.agents.iter_mut().enumerate() {
                let act = agent.action(&mut self.env);
                let (status, reward) = self.env.step(act);
                agent.push_result(reward);

                if status.is_terminal() {
                    for x in 0..i {
                        self.agents[x].end_episode(&self.env, E::LOSS);
                    }

                    self.agents[i].end_episode(&self.env, 0.);
                    
                    for x in i + 1..self.agents.len() {
                        self.agents[x].end_episode(&self.env, E::LOSS);
                    }
                    return status;
                }
            }
        }
    }
}

pub trait AgentWrapper {
    type Env: Enviroment;

    fn action(&mut self, env: &Self::Env) -> <Self::Env as Enviroment>::Action;

    fn begin_episode(&mut self, env: &Self::Env);

    fn push_result(&mut self, reward: f32);

    fn end_episode(&mut self, env: &Self::Env, reward: f32);
}

pub struct Wrapper<A: Agent> {
    agent: A,
    data: A::Data,
    reward: Option<f32>,
    action: Option<<A::Env as Enviroment>::Action>,
}

impl<A: Agent> Wrapper<A> {
    pub fn new(agent: A, data: A::Data) -> Self {
        Self {
            agent,
            data,
            reward: None,
            action: None,
        }
    }
}

impl<A: Agent> AgentWrapper for Wrapper<A>
where
    <A::Env as Enviroment>::Action: Clone,
{
    type Env = A::Env;

    fn action(&mut self, env: &Self::Env) -> <Self::Env as Enviroment>::Action {
        if let Some(reward) = self.reward.take() {
            if let Some(action) = self.action.take() {
                self.data.push_result(env, action, reward);
                self.agent.update(&self.data);
            }
        }

        let act = self.agent.action(env, &self.data);
        self.action.replace(act.clone());
        act
    }

    fn begin_episode(&mut self, env: &Self::Env) {
        self.reward.take();
        self.action.take();
        self.data.begin_episode(env);
    }

    fn push_result(&mut self, reward: f32) {
        self.reward.replace(reward);
    }

    fn end_episode(&mut self, env: &Self::Env, reward: f32) {
        let action = self.action.take().expect("Cached action missing");
        let reward = self.reward.expect("Cached reward missing") + reward;
        self.data.push_result(env, action, reward);
        self.agent.update(&self.data);
    }
}
