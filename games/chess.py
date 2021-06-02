import datetime
import os
from abc import ABC

import gym
import gym_chess
import numpy as np
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None

        ### Game
        self.observation_shape = (8, 8, 119)
        self.action_space = list(range(4672))
        self.players = list(range(2))
        self.stacked_observations = 0  # 100 in chess we increased the history to the last 100 board states to allow correct prediction of draws

        self.muzero_player = 0
        self.opponent = "expert"

        ### Self-play
        self.num_workers = 1
        self.selfplay_on_gpu = True
        self.max_moves = 50  # 512 based on pseudocode
        self.num_simulations = 1  # based on paper
        self.discount = 1  # based on pseudocode
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"
        self.support_size = 10

        # Residual Network
        self.downsample = False
        self.blocks = 1
        self.channels = 16
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        # Path to store the model weights and TensorBoard logs
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100  # 200K Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.1  # Initial learning rate
        self.lr_decay_rate = 0.1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 3000  # 3000 Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    @staticmethod
    def visit_softmax_temperature_fn(trained_steps):
        if trained_steps < 30:
            return 1.0
        else:
            return 0.0


class Game(AbstractGame, ABC):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.env = gym.make("ChessAlphaZero-v0")
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        return np.array(observation), reward, done

    def legal_actions(self):
        return self.env.legal_actions

    def reset(self):
        return np.array(self.env.reset())

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
        input("Press Enter to take a step ")

    def action_to_string(self, action_number):
        return self.env.decode(action_number)
