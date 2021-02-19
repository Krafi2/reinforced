pub trait Enviroment {
    type Action;
}

pub trait Encode {
    fn encode(&self) -> &[f32];
}

pub enum Message<'a, E: Enviroment> {
    StartEpisode(&'a E),
    Action(E::Action),
    Reward(f32),
    State(&'a E),
    EndEpisode(&'a E),
}


pub struct Discrete(pub u32);