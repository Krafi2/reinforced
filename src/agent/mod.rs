pub mod policy;
pub mod handler;
pub mod agents;

pub use self::handler::Handler;
pub use self::policy::Policy;

pub trait Agent<M, E>: Handler<M> + Policy<Env = E> {}
impl<T, M, E> Agent<M, E> for T where T: Handler<M> + Policy<Env = E> {}