pub trait Enviroment {
    type Action;
    type Status;

    fn reset(&mut self);
    fn step(&mut self, action: Self::Action) -> (Self::Status, f32);
    fn validate(&self, action: Self::Action) -> bool;
}

pub trait EnvBuilder {
    type Output: Enviroment;

    fn build(self) -> Self::Output;
}

pub trait GetToken {
    type Token;

    fn get_token(&mut self) -> Self::Token;
}

pub trait PlayerRange {
    const MIN: usize;
    const MAX: Option<usize>;
}

pub trait IsTerminal {
    fn is_terminal(&self) -> bool;
}

/// This trait indicates that an enviroment can have only a single winner
/// And all other participating agents will receive the reward of `LOSS`
pub trait SingleWinner {
    const LOSS: f32;
}

pub mod discrete {
    #[derive(Copy, Clone)]
    pub struct TaggedDiscrete {
        pub action: u32,
        pub player: u32,
    }

    #[derive(Clone)]
    pub struct ActionToken {
        player: u32,
        max: u32,
    }

    impl ActionToken {
        pub fn action(&self, action: u32) -> Option<TaggedDiscrete> {
            if action > self.max {
                Some(TaggedDiscrete {
                    action,
                    player: self.player,
                })
            } else {
                None
            }
        }
    }

    impl IntoIterator for &ActionToken {
        type Item = TaggedDiscrete;

        type IntoIter = Iter;

        fn into_iter(self) -> Self::IntoIter {
            Iter {
                token: self.clone(),
                idx: 0,
            }
        }
    }

    pub struct Iter {
        token: ActionToken,
        idx: u32,
    }

    impl Iterator for Iter {
        type Item = TaggedDiscrete;

        fn next(&mut self) -> Option<Self::Item> {
            if self.idx <= self.token.max {
                let res = self.token.action(self.idx);
                self.idx += 1;
                res
            } else {
                None
            }
        }
    }
}
