pub mod agents;
pub mod handler;
pub mod policy;

pub use self::agents::q_agent::{self, QAgent};
pub use self::handler::{
    replay_buffer::{DataPoint, ReplayBuffer, Transition},
    Handler,
};
pub use self::policy::Policy;

use crate::enviroment::Message;

pub trait Agent<M, E>: Handler<M> + Policy<Env = E> {}

impl<T, M, E> Agent<M, E> for T where T: Handler<M> + Policy<Env = E> {}

pub trait MessageAgent<E>
where
    for<'a> Self: Agent<Message<'a, E>, E>,
{
}

impl<T, E> MessageAgent<E> for T where for<'a> T: Agent<Message<'a, E>, E> {}
