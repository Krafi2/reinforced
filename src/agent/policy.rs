use crate::enviroment::Enviroment;

pub trait Policy {
    type Env: Enviroment;

    fn action(&mut self, env: &Self::Env) -> <Self::Env as Enviroment>::Action;
}