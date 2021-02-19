mod tic_tac_toe;
use std::env;

use tic_tac_toe::{ActionResult, EnvManager, StepResult, TicTacToe};

use reinforced::agent::{
    q_agent::{bellman, epsilon, QBuilder, QTrainer},
    QAgent,
};
use rusty_nn::{
    a_funcs::{Identity, TanH},
    initializer::{Ones, Xavier},
    layers::{map_layer::MapLayer, BasicLayer, DenseBuilder, MapBuilder},
    loss::MeanSquared,
    network::{FeedForward, LinearBuilder},
    optimizer::{Adam, AdamBuilder, DefaultOptimizer, GradientDescent, OptimizerBuilder},
    trainer::{LogFile, MockLogger, Trainer},
};

fn main() {
    let mut args = env::args();
    let print = match args.nth(1).as_ref().map(String::as_str) {
        Some("-p") => true,
        _ => false,
    };

    // color_backtrace::install();

    let width = 3;
    let height = 3;
    let chain = 3;
    let loss_reward = -1.;
    let win_reward = 1.;
    let tie_reward = 0.5;

    let batch_size = 100;
    let epochs = 1;
    let epsilon = 0.1;
    let l_rate = 0.01;

    let discount = 0.9;
    let memory = 200;
    let train_every = 1;
    let lag = 10;
    let epsilon_greedy = epsilon::Static::new(discount);
    let q_func = bellman::SoftBellman::new(0.9);

    let agent = |logger| {
        let network = LinearBuilder::<BasicLayer>::new(27)
            .layer(DenseBuilder::new(TanH, Xavier::new(), 27, true, true))
            .layer(DenseBuilder::new(TanH, Xavier::new(), 9, true, true))
            .layer(MapBuilder::new(Identity, Ones))
            .build::<FeedForward>()
            .unwrap();

        let optimizer = OptimizerBuilder::new()
            .network(network)
            .optimizer(AdamBuilder::new().l_rate(l_rate).epsilon(epsilon).build())
            .loss(MeanSquared)
            .build()
            .unwrap();

        QBuilder::new()
            .optimizer(optimizer)
            .memory(memory)
            .train_every(train_every)
            .lag(lag)
            .trainer(Box::new(QTrainer::new(batch_size, epochs, 0, logger)))
            .q_target(Box::new(q_func.clone()))
            .eps(Box::new(epsilon_greedy.clone()))
            .build()
            .unwrap()
    };

    let agent1 = agent(Box::new(LogFile::new("log.txt").unwrap()));
    let agent2 = agent(Box::new(MockLogger));

    let mut env = EnvManager::new(
        agent1,
        agent2,
        width,
        height,
        chain,
        loss_reward,
        win_reward,
        tie_reward,
    );

    for e in 0..10_000 {
        let mut episode = env.run();
        let print = print && (e % 200) == 0;

        if print {
            println!("{}", episode);
        }

        while let Some(ActionResult {
            player,
            action,
            result,
        }) = episode.next()
        {
            if print {
                if matches!(
                    result,
                    StepResult::Win | StepResult::Tie | StepResult::Continue
                ) {
                    println!("Player {} played action {}", player, action);
                    println!("{}", episode);
                }

                match result {
                    StepResult::Win => {
                        println!("=============================================");
                        println!("Player {} has won the game", player);
                        println!("=============================================\n");
                    }
                    StepResult::Tie => {
                        println!("=============================================");
                        println!("It's a tie");
                        println!("=============================================\n");
                    }
                    StepResult::InvalidAction => {
                        println!("=============================================");
                        println!(
                            "Player {} tried to play an invalid action {}",
                            player, action
                        );
                        println!("=============================================\n");
                    }
                    StepResult::Continue => {}
                }
            }
        }
    }
}
