use std::{
    convert::TryFrom,
    fmt::{self, Display},
    iter::Enumerate,
    vec,
};

use reinforced::{
    agent::MessageAgent,
    enviroment::{Discrete, Encode, Message, Status},
    Enviroment, Handler, Policy,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    Cross,
    Circle,
    Empty,
}

impl Into<u32> for Cell {
    fn into(self) -> u32 {
        match self {
            Cell::Cross => 0,
            Cell::Circle => 1,
            Cell::Empty => 2,
        }
    }
}

impl TryFrom<u32> for Cell {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Cell::Cross),
            1 => Ok(Cell::Circle),
            2 => Ok(Cell::Empty),
            _ => Err(()),
        }
    }
}

impl Cell {
    pub fn as_num(&self) -> u32 {
        (*self).into()
    }

    pub fn from_num(num: u32) -> Result<Self, ()> {
        Self::try_from(num)
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            _ => false,
        }
    }
}

use board::Board;
mod board {
    use std::fmt::Write;

    use super::*;

    type Result<T> = std::result::Result<T, ()>;

    #[derive(Debug, Clone)]
    pub struct Board {
        width: u32,
        height: u32,
        filled: u32,
        board: Vec<Cell>,
        one_hot: Vec<f32>,
    }

    impl Board {
        pub(super) fn new(width: u32, height: u32) -> Self {
            let len = (width * height) as usize;
            Self {
                width,
                height,
                filled: 0,
                board: vec![Cell::Empty; len],
                one_hot: {
                    let mut vec = vec![0.; len * 3];
                    for i in &mut vec[len * 2..] {
                        *i = 1.
                    }
                    vec
                },
            }
        }

        /// Get the board's width.
        pub fn width(&self) -> u32 {
            self.width
        }

        /// Get the board's height.
        pub fn height(&self) -> u32 {
            self.height
        }

        pub(super) fn reset(&mut self) {
            self.board.fill(Cell::Empty);
            self.one_hot.fill(0.);
            for i in &mut self.one_hot[(self.width * self.height) as usize * 2..] {
                *i = 1.
            }
            self.filled = 0;
        }

        pub fn one_hot(&self) -> &[f32] {
            &self.one_hot
        }

        pub fn coords_to_idx(&self, x: u32, y: u32) -> Result<u32> {
            if x < self.width && y < self.height {
                Ok(y * self.width + x)
            } else {
                Err(())
            }
        }

        pub fn idx_to_coords(&self, idx: u32) -> Result<(u32, u32)> {
            if idx < self.width * self.height {
                Ok((idx % self.width, idx / self.width))
            } else {
                Err(())
            }
        }

        pub(super) fn xy(&self, x: u32, y: u32) -> Result<Cell> {
            let idx = self.coords_to_idx(x, y)?;
            Ok(self.board[idx as usize])
        }

        pub(super) fn set_xy(&mut self, x: u32, y: u32, cell: Cell) -> Result<()> {
            self.set_at_unchecked(self.coords_to_idx(x, y)?, cell);
            Ok(())
        }

        pub(super) fn at(&self, idx: u32) -> Result<Cell> {
            if idx < self.width * self.height {
                Ok(self.board[idx as usize])
            } else {
                Err(())
            }
        }

        fn set_at_unchecked(&mut self, idx: u32, cell: Cell) {
            let old = self.board[idx as usize];

            // This will increment filled if the cell was previously empty and now is not, decrement it in the opposite case
            // and leave the value unchanged in all other cases
            self.filled += (!cell.is_empty()) as u32 - (!old.is_empty()) as u32;

            let len = self.width * self.height;
            self.one_hot[(len * old.as_num() + idx) as usize] = 0.;
            self.one_hot[(len * cell.as_num() + idx) as usize] = 1.;
            self.board[idx as usize] = cell;
        }

        pub(super) fn set_at(&mut self, idx: u32, cell: Cell) -> Result<()> {
            let len = self.width * self.height;
            if idx < len {
                self.set_at_unchecked(idx, cell);
                Ok(())
            } else {
                Err(())
            }
        }

        pub fn is_full(&self) -> bool {
            self.filled == self.width * self.height
        }

        pub fn filled(&self) -> u32 {
            self.filled
        }

        pub(super) fn iter<'a>(&'a self) -> std::slice::Iter<'a, Cell> {
            self.board.iter()
        }
    }

    impl Display for Board {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let len = self.width * self.height;
            let n_len = f32::log10((len - 1) as f32).ceil() as usize;

            for y in 0..self.height {
                for x in 0..self.height {
                    let idx = y * self.width + x;
                    let str = match self.at(idx).expect("Index out of bounds") {
                        Cell::Cross => "X".to_owned(),
                        Cell::Circle => "O".to_owned(),
                        Cell::Empty => "_".to_owned(),
                        // Cell::Empty => idx.to_string(),
                    };
                    write!(f, "{:1$} ", str, n_len)?;
                }
                f.write_char('\n')?;
            }
            Ok(())
        }
    }
}

use player_id::PlayerId;
mod player_id {
    use super::*;
    #[derive(Clone, Copy, Debug)]
    pub(super) struct PlayerId(u32);

    impl Display for PlayerId {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt(f)
        }
    }

    impl PlayerId {
        pub(super) fn next(&self) -> PlayerId {
            Self(1 - self.0)
        }
    }

    impl Default for PlayerId {
        fn default() -> Self {
            Self(0)
        }
    }

    impl Into<usize> for PlayerId {
        fn into(self) -> usize {
            self.0 as usize
        }
    }

    impl Into<Cell> for PlayerId {
        fn into(self) -> Cell {
            Cell::try_from(self.0).expect("Somehow I am larger than 1")
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ActionResult {
    pub player: u32,
    pub action: Discrete,
    pub result: StepResult,
}

#[derive(Clone, Copy, Debug)]
pub enum StepResult {
    Win,
    Tie,
    InvalidAction,
    Continue,
}

pub use tic_tac_toe::TicTacToe;
pub mod tic_tac_toe {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct TicTacToe {
        board: Board,
        to_play: PlayerId,
        status: Status,
        chain: u32,
    }

    impl TicTacToe {
        pub(super) fn new(width: u32, height: u32, chain: u32) -> Self {
            Self {
                board: Board::new(width, height),
                to_play: PlayerId::default(),
                status: Status::Start,
                chain,
            }
        }

        pub(super) fn step(&mut self, action: Discrete) -> StepResult {
            let idx = action.0;
            let player = self.to_play().into();

            if self.board.at(idx).unwrap_or(Cell::Circle).is_empty() {
                self.board.set_at(idx, player).unwrap();

                const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 1), (1, 0), (1, -1)];
                let (x, y) = self.board.idx_to_coords(idx).expect("Index out of bounds");
                for (dx, dy) in DIRECTIONS.iter().copied() {
                    let mut accumulator = 1;
                    for k in [1, -1].iter().copied() {
                        let mut x = x as i32;
                        let mut y = y as i32;
                        loop {
                            x += k * dx;
                            y += k * dy;

                            match self.board.xy(x as u32, y as u32) {
                                Ok(cell) => match cell == player {
                                    true => accumulator += 1,
                                    false => break,
                                },
                                Err(_) => break,
                            };
                        }
                    }

                    if accumulator >= self.chain {
                        return StepResult::Win;
                    }
                }

                match self.board.is_full() {
                    true => {
                        self.status = Status::End;
                        StepResult::Tie
                    }
                    false => {
                        if self.board.filled() > 1 {
                            self.status = Status::Playing;
                        }
                        StepResult::Continue
                    }
                }
            } else {
                StepResult::InvalidAction
            }
        }

        pub(super) fn reset(&mut self) {
            self.board.reset();
            self.to_play = PlayerId::default();
            self.status = Status::Start;
        }

        pub(super) fn to_play(&self) -> PlayerId {
            self.to_play
        }

        pub(super) fn next_player(&mut self) {
            self.to_play = self.to_play.next();
        }
    }

    impl Enviroment for TicTacToe {
        type Action = Discrete;
        type Status = Status;

        fn status(&self) -> Self::Status {
            self.status
        }

        fn actions(&self) -> Vec<Self::Action> {
            self.board
                .iter()
                .enumerate()
                .filter_map(|(i, cell)| cell.is_empty().then(|| Discrete::new(i as u32)))
                .collect()
        }
    }

    impl Encode for TicTacToe {
        fn encode(&self) -> &[f32] {
            self.board.one_hot()
        }
    }

    impl Display for TicTacToe {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.board.fmt(f)
        }
    }
}

pub struct EnvManager {
    env: TicTacToe,
    players: [Box<dyn MessageAgent<TicTacToe>>; 2],

    loss_reward: f32,
    win_reward: f32,
    tie_reward: f32,
}

impl EnvManager {
    pub fn new<P1, P2>(
        agent1: P1,
        agent2: P2,
        width: u32,
        height: u32,
        chain: u32,
        loss_reward: f32,
        win_reward: f32,
        tie_reward: f32,
    ) -> Self
    where
        P1: MessageAgent<TicTacToe> + 'static,
        P2: MessageAgent<TicTacToe> + 'static,
    {
        Self {
            env: TicTacToe::new(width, height, chain),
            players: [
                Box::new(agent1) as Box<dyn MessageAgent<TicTacToe>>,
                Box::new(agent2) as Box<dyn MessageAgent<TicTacToe>>,
            ],
            loss_reward,
            win_reward,
            tie_reward,
        }
    }

    pub fn run<'a>(&'a mut self) -> Episode<'a> {
        Episode::new(self)
    }

    pub fn reset(&mut self) {
        self.env.reset();
    }

    pub fn env(&self) -> &TicTacToe {
        &self.env
    }
}

pub struct Episode<'a> {
    inner: &'a mut EnvManager,
    status: StepResult,
}

impl<'a> Display for Episode<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.env.fmt(f)
    }
}

impl<'a> Episode<'a> {
    fn new(env: &'a mut EnvManager) -> Episode<'a> {
        env.reset();
        Self {
            inner: env,
            status: StepResult::Continue,
        }
    }

    pub fn step(&mut self) -> ActionResult {
        let inner = &mut self.inner;
        let id = inner.env.to_play();
        let player = inner.players[<PlayerId as Into<usize>>::into(id)].as_mut();
        player.handle(Message::State(&inner.env));
        let action = player.action(&inner.env);

        let result = inner.env.step(action);
        match result {
            StepResult::Win => {
                player.handle(Message::Action(action));
                player.handle(Message::Reward(inner.win_reward));
                player.handle(Message::State(&inner.env));
                let other = inner.players[<PlayerId as Into<usize>>::into(id.next())].as_mut();
                other.handle(Message::Reward(inner.loss_reward));
                other.handle(Message::State(&inner.env));
            }
            StepResult::Tie => {
                player.handle(Message::Action(action));
                player.handle(Message::Reward(inner.tie_reward));
                player.handle(Message::State(&inner.env));
                let other = inner.players[<PlayerId as Into<usize>>::into(id.next())].as_mut();
                other.handle(Message::Reward(inner.tie_reward));
                other.handle(Message::State(&inner.env));
            }
            StepResult::InvalidAction => {
                player.handle(Message::InvalidAction(action));
            }
            StepResult::Continue => {
                player.handle(Message::Action(action));
                player.handle(Message::Reward(inner.tie_reward));
                inner.env.next_player();
            }
        }
        self.status = result;

        ActionResult {
            player: <PlayerId as Into<usize>>::into(id) as u32,
            action,
            result,
        }
    }
}

impl<'a> Iterator for Episode<'a> {
    type Item = ActionResult;

    fn next(&mut self) -> Option<Self::Item> {
        match self.status {
            StepResult::Win | StepResult::Tie | StepResult::InvalidAction => None,
            StepResult::Continue => Some(self.step()),
        }
    }
}
