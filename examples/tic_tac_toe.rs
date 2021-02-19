use reinforced::{Enviroment, Handler, Policy, Agent, enviroment::{Message, Discrete, Encode}};

#[derive(Clone, Copy)]
pub enum Cell {
    Cross,
    Circle,
    Empty
}

impl Cell {
    pub fn as_num(&self) -> usize {
        match self {
            Cell::Cross => 0,
            Cell::Circle => 1,
            Cell::Empty => 2,
        }
    }
}

use board::Board;
mod board {
    use super::*;

    pub struct Board {
        width: usize,
        height: usize,
        filled: usize,
        board: Vec<Cell>,
        one_hot: Vec<f32>,
    }
    
    impl Board {
        pub (super) fn new(width: usize, height: usize) -> Self {
            let len = width * height;
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
        pub fn width(&self) -> usize {
            self.width
        }
    
        /// Get the board's height.
        pub fn height(&self) -> usize {
            self.height
        }
    
        pub (super) fn reset(&mut self) {
            self.board.fill(Cell::Empty);
            self.one_hot.fill(0.);
            for i in &mut self.one_hot[self.width * self.height * 2..] {
                *i = 1.
            }
        }

        pub fn one_hot(&self) -> &[f32] {
            &self.one_hot
        }

        fn coords_to_idx(&self, x: usize, y: usize) -> usize {
            assert!(x < self.width);
            assert!(y < self.height);
            y * self.width + x
        }
    
        fn idx_to_coords(&self, idx: usize) -> (usize, usize) {
            assert!(idx < self.width * self.height);
            (idx % self.width, idx / self.width)
        } 
    
        pub (super) fn xy(&self, x: usize, y: usize) -> Cell {
            let idx = self.coords_to_idx(x, y);
            self.board[idx]
        }
    
        fn xy_mut(&mut self, x: usize, y: usize) -> &mut Cell {
            let idx = self.coords_to_idx(x, y);
            &mut self.board[idx]
        }
    
        pub (super) fn set_xy(&mut self, x: usize, y: usize, cell: Cell) {
            self.set_at(self.coords_to_idx(x, y), cell);
        }
        
        pub (super) fn at(&self, idx: usize) -> Cell {
            assert!(idx < self.width * self.height);
            self.board[idx]
        }
    
        fn at_mut(&mut self, idx: usize) -> &mut Cell {
            assert!(idx < self.width * self.height);
            &mut self.board[idx]
        }
    
        pub (super) fn set_at(&mut self, idx: usize, cell: Cell) {
            let len = self.width * self.height;
            assert!(idx < len);
            let old = self.board[idx];
            self.one_hot[len * old.as_num() + idx] = 0.;
            *self.at_mut(idx) = cell;
            self.one_hot[len * cell.as_num() + idx] = 1.;
        }
    }
}


pub struct TicTacToe<'a> {
    board: Board,
    players: [Box<dyn Agent<Message<'a, Self>, Self>>; 2],
}

impl<'a> Enviroment for TicTacToe<'a> {
    type Action = Discrete;
}

impl<'a> Encode for TicTacToe<'a> {
    fn encode(&self) -> &[f32] {
        self.board.encode();
    }
}

fn main() {

}