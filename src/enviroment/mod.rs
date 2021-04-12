use std::fmt::Display;

pub trait Enviroment {
    type Status;
    type Action;

    fn status(&self) -> Self::Status;
    fn actions(&self) -> Vec<Self::Action>;
}

pub trait Encode {
    // TODO make this a Cow
    fn encode(&self) -> &[f32];
}

pub trait IsTerminal {
    fn it_terminal(&self) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub enum Status {
    Start,
    Playing,
    End,
}

impl IsTerminal for Status {
    fn it_terminal(&self) -> bool {
        match self {
            Status::End => true,
            _ => false,
        }
    }
}

pub enum Message<'a, E: Enviroment> {
    Action(E::Action),
    InvalidAction(E::Action),
    Reward(f32),
    State(&'a E),
}

#[derive(Clone, Copy, Debug)]
pub struct Discrete(pub u32);

impl Display for Discrete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Discrete {
    pub fn new(act: u32) -> Self {
        Self(act)
    }

    pub fn inner(&self) -> u32 {
        self.0
    }
}
