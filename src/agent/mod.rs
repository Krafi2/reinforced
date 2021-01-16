use crate::enviroment::Enviroment;
use crate::manager::data_collector::DataCollector;

// pub mod adapter;
pub mod learning;

/// This trait is used to querry agents for an action.
pub trait Agent {
    type Env: Enviroment;
    type Data: DataCollector<Env = Self::Env>;

    fn action(&mut self, env: &Self::Env, data: &Self::Data) -> <Self::Env as Enviroment>::Action;
    fn update(&mut self, data: &Self::Data);
}

pub trait AgentBuilder<E> {
    type Data: DataCollector;
    type Agent: Agent<Data = Self::Data>;

    fn build(self, env: &mut E) -> (Self::Agent, Self::Data);
}
