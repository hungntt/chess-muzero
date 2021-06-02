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
        return

    @staticmethod
    def visit_softmax_temperature_fn(trained_steps):
        return


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


class Officers:
    def __init__(self, num_officer, num_event, num_task):
        # Total number of officer
        self.num_officer = num_officer
        self.num_event = num_event  # Total number of event
        self.num_task = num_task  # Total number of task per each event

        # initialize capability matrix for each event, each task
        self.capability = self.random_capability()
        # self.capability = example.officer_capability

    def random_capability(self):
        officers = []
        # Assign random capability for each officer
        for i in range(self.num_officer):
            capability = np.random.randint(1, 20, (self.num_event, self.num_task))
            officers.append(capability)
        return officers

    def reset_capability(self):
        self.capability = self.random_capability()


class Events:
    def __init__(self, num_event, num_task):
        self.num_event = num_event
        self.num_task = num_task
        self.num_problems = num_event * num_task
        # Initialize when each event start
        self.occurrence = self.random_occurrence()
        # self.occurrence = example.occurrence
        # Initialize transition matrix from base station to each event
        self.transition_matrix = self.random_transition()
        # self.transition_matrix = example.transition_matrix

    def random_occurrence(self):
        occurrence = np.random.randint(1, 20, self.num_event)
        occurrence = np.sort(occurrence)
        return occurrence

    def random_transition(self):
        n = self.num_event + 1
        # Return a symmetric matrix with 0 in the diagonal
        distribution = np.random.randint(1, 20, int(n * (n - 1) / 2))
        matrix = np.zeros((n, n))
        matrix[np.triu_indices(n, 1)] = distribution
        matrix[np.tril_indices(n, -1)] = matrix.T[np.tril_indices(n, -1)]
        return matrix

    def reset_events(self):
        self.occurrence = self.random_occurrence()
        self.transition_matrix = self.random_transition()


class EventsWithPriority(Events):
    def __init__(self, num_event, num_task, priority_type="score"):
        super().__init__(num_event, num_task)
        self.priority_type = priority_type
        # Introduce priority for each event
        if priority_type == "rank":
            # the higher the number, the higher the priority
            self.set_rank_priority(priority_list="random")
        elif priority_type == "score":
            # the highest score the
            self.set_score_priority()
        else:
            raise Exception("Unknown priority type")

    def set_rank_priority(self, priority_list):
        if priority_list == "random":
            self.priority = np.random.choice(range(1, self.num_event + 1), size=self.num_event, replace=False)
        else:
            if len(priority_list) != self.num_event:
                raise Exception("Priority list must have the same length with number of events")
            self.priority = priority_list

    def set_score_priority(self, priority_list=None):
        if priority_list == "random":
            self.priority = np.random.randint(0, 100, size=self.num_event)
        else:
            if len(priority_list) != self.num_event:
                raise Exception("Priority list must have the same length with number of events")
            self.priority = priority_list

