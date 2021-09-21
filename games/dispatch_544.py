import datetime
import os
from abc import ABC

import numpy as np
import ray
import torch

from .abstract_game import AbstractGame

IDLE = 1
RUNNING = 2
COMPLETED = 3

NUM_OFFICER = 5
NUM_TASK = 4
NUM_EVENT = 4


# def input_config():
#     ### Input config for Dispatch
#     args = DotDict({
#         'num_officer': int(input("O: ")),
#         'num_event': int(input("E: ")),
#         'num_task': int(input("T: ")),
#     })
#     return args


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
        # args = input_config()
        self.method = 0
        self.officer = Officers(NUM_OFFICER, NUM_EVENT, NUM_TASK)
        self.event = Events(NUM_EVENT, NUM_TASK)
        self.env = Dispatch(officers=self.officer, events=self.event)

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = self.env.get_observation().shape  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(NUM_OFFICER))
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = NUM_OFFICER * NUM_TASK * NUM_EVENT  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 20  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 4  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network

        ### Training
        self.folder_path = f'{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}' \
                           f'{NUM_OFFICER}O-{NUM_EVENT}E-{NUM_TASK}T'
        # self.folder_random_path = f'RANDOM_{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}' \
        #                           f'{NUM_OFFICER}O-{NUM_EVENT}E-{NUM_TASK}T'
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3],
                                         self.folder_path)  # Path to store the model weights and TensorBoard logs
        # self.random_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
        #                                 os.path.basename(__file__)[:-3], self.folder_random_path)
        # self.arena_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../arena",
        #                                  os.path.basename(__file__)[:-3], self.folder_path)
        self.logger_path = os.path.join(self.results_path, "./logger.txt")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 350000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 5000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 25  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        return 1


class Game(AbstractGame, ABC):
    def __init__(self, seed=None):
        # self.args = input_config()
        self.officer = Officers(NUM_OFFICER, NUM_EVENT, NUM_TASK)
        self.event = Events(NUM_EVENT, NUM_TASK)
        self.env = Dispatch(officers=self.officer, events=self.event)

    def step(self, action):
        observation, reward, done = self.env.step(action)
        return np.array(observation), reward, done

    def legal_actions(self):
        legal_actions = self.env.legal_actions()
        return legal_actions

    def reset(self):
        return np.array(self.env.reset())

    def render(self):
        self.env.render()
        input("Press Enter to take a step ")


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


class Dispatch:
    def __init__(self, officers, events):
        self.officers = officers
        self.events = events
        self.num_problems = events.num_problems
        # self.actual_capability = self.random_capability() # add noise to officers' capability
        self.actual_capability = self.officers.capability  # No noise
        self.reset(reset_officers=False, reset_events=False)
        self.action_table = np.full(self.events.num_task * self.events.num_event, -1, dtype="int32")
        self.obs = self.get_observation()
        self.len_obs = None
        self.event_time_end = []
        self.event_time_completed = []
        ### For reset()
        self.officer_occupied = [False] * self.officers.num_officer
        # codes Idle = 1, Running = 3, Over = 2
        # These are created for initialization and comparison when all tasks are done
        self.event_task_status_start = np.full((self.events.num_event, self.events.num_task), IDLE)
        self.event_task_status_end = np.full((self.events.num_event, self.events.num_task), COMPLETED)

        # To record the actual time of each task
        self.event_task_time_start = np.zeros((self.events.num_event, self.events.num_task))
        self.event_task_time_end = np.zeros((self.events.num_event, self.events.num_task))

        # In each event which task (task ID) is running. Position is event ID and value is Task ID
        self.running_task = [0] * self.events.num_event

        self.max_time = np.max(np.array(self.officers.capability)) * self.events.num_problems + \
                        np.max(np.array(self.events.transition_matrix)) * self.num_problems

    def random_capability(self):
        actual_capability = []
        # Introduce gaussian randomness into the actual time taken
        for cap in self.officers.capability:
            mu, sigma = 0, 0.1
            add_ = np.random.normal(mu, sigma, cap.shape)
            actual_capability.append(add_ + cap)
        return actual_capability

    def reset_actual_capability(self):
        self.actual_capability = self.random_capability()

    def reset(self, reset_officers=True, reset_events=True):
        # reset officers
        self.action_table = np.full(self.events.num_task * self.events.num_event, -1, dtype="int32")
        if reset_officers:
            self.officers.reset_capability()

        # reset events
        if reset_events:
            self.events.reset_events()
            self.obs = self.get_observation()  # Update observation

        return self.get_observation()

    def reset_action_table(self):
        self.action_table = np.full(self.events.num_task * self.events.num_event, -1, dtype="int32")

    def step(self, assignment):
        #  Simulation
        # Input: assignment of the officers is a matrix
        #        occurrence (time start of each event) from attribute
        #        actual capability from attribute
        #        transition matrix (time unit to move) from attribute

        # Step 1: Initialization
        # Each event will have a list/dictionary
        # [Task_number, start_time, end_time, officer_assigned]
        # [-1, -1, -1, -1] -> initialization value
        if type(assignment) == int or isinstance(assignment, (int, np.integer)):
            assignment = self.action_to_matrix(assignment)

        event_list = []
        for i in range(self.events.num_event):
            event_list.append({"task_id": -1, "start_time": -1, "end_time": -1, "officer_assigned": -1})

        # book-keeping which officer is working [False, False, False]
        officer_occupied = [False] * self.officers.num_officer

        # book-keeping officer location: -1: Base-station, 0:event_0, ... [-1, -1, -1]
        officer_location = [-1] * self.officers.num_officer

        # keep track of status of event_task_status:
        # array([[1, 1],
        #        [1, 1]])
        event_task_status = self.event_task_status_start.copy()

        # get the sequence of event will happen occurrence = {0: 7, 1: 8}
        occurrence = {}
        for i, occur_time in enumerate(self.events.occurrence):
            occurrence[i] = occur_time

        # book-keeping for end-time of event end_time = {0: -1, 1: -1}
        end_time = {}
        for i in range(len(occurrence)):
            end_time[i] = -1

        # Step 2: Simulation
        check_end = np.sum(self.event_task_status_end - event_task_status)
        time = 0
        tie = False

        # repeat until all tasks are done
        while check_end > 0:
            # # Debug
            # print("Time:", time)
            # for i in range(len(event_list)):
            #     print("Event {}: {}".format(i, event_list[i]))
            # print("End time list: {}".format(end_time))
            # print("Check end: {}".format(check_end))
            # print("Event task status: {}".format(event_task_status))

            # Filter events > current time in both occurrence and end_time
            next_events_time = []
            next_events_indices = []
            for key, value in occurrence.items():
                if value >= time:
                    next_events_time.append(value)
                    next_events_indices.append(key)

            for key, value in end_time.items():
                if value >= time:
                    next_events_time.append(value)
                    next_events_indices.append(key)

            assert (len(next_events_time) > 0)
            time = np.min(next_events_time)

            # Choose minimum from the filtered elements and extract Event ID.
            min_event_time = []
            min_event_indices = []
            for key, value in occurrence.items():
                if value == time:
                    min_event_time.append(value)
                    min_event_indices.append(key)

            for key, value in end_time.items():
                if value == time:
                    min_event_time.append(value)
                    min_event_indices.append(key)

            # In the case that's there are more than 2 events with the same occurring time
            if len(min_event_time) > 1:
                tie = True
                min_event_index = self.get_event_tie(min_event_indices, event_list, event_task_status)
            else:
                tie = False
                min_event_index = min_event_indices[0]  # remember that events already sorted with time occurrence

            # If Event(EventID) has not yet started, all tasks will have the status == 1 -> idle
            if event_task_status[min_event_index, 0] == IDLE:
                officer_assigned = assignment[min_event_index, 0]  # first task of the event
                task_id = event_list[min_event_index]["task_id"]
            # if Event(EventID) has started and is not in last task:
            elif event_task_status[min_event_index, -1] == IDLE:
                task_id = event_list[min_event_index]["task_id"]
                officer_assigned = assignment[min_event_index, task_id + 1]  # subsequent task
            else:
                officer_id = event_list[min_event_index]["officer_assigned"]
                officer_occupied[officer_id] = False
                del end_time[min_event_index]
                task_id = event_list[min_event_index]["task_id"]
                event_task_status[min_event_index, task_id] = COMPLETED
                check_end = np.sum(self.event_task_status_end - event_task_status)
                continue

            # Check officer occupied for Officer ID.
            if not officer_occupied[officer_assigned]:
                # If Event(EventID) has not yet started
                if event_task_status[min_event_index, 0] == IDLE:
                    # Remove occurrence of Event(EventID) from List1
                    del occurrence[min_event_index]
                    # Extract Officer ID location : Loc
                    location = officer_location[officer_assigned]
                    # Update the data
                    event_list[min_event_index]["task_id"] = task_id + 1
                    event_list[min_event_index]["start_time"] = time
                    working_time = self.actual_capability[officer_assigned][min_event_index, task_id + 1]
                    travelling_time = self.events.transition_matrix[location, min_event_index]
                    event_list[min_event_index]["end_time"] = time + working_time + travelling_time
                    event_list[min_event_index]["officer_assigned"] = officer_assigned
                    end_time[min_event_index] = event_list[min_event_index]["end_time"]
                    # Set Officer Occupied (Officer ID) = TRUE
                    officer_occupied[officer_assigned] = True
                    # Set Officer Location (Officer ID) = Event(EventID) Location
                    officer_location[officer_assigned] = min_event_index
                    # set event_task_status(EventId, Task number) = running
                    event_task_status[min_event_index, task_id + 1] = RUNNING
                # If event has been running
                else:
                    if event_list[min_event_index]["officer_assigned"] != -2:
                        previous_officer = event_list[min_event_index]["officer_assigned"]
                        # Set Officer Occupied (Previous_officer ID) = FALSE
                        officer_occupied[previous_officer] = False
                        # set event_task_status(EventID, Task number) = Completed
                        event_task_status[min_event_index, task_id] = COMPLETED

                    # Extract Officer ID location : Loc
                    location = officer_location[officer_assigned]
                    # Update the data
                    event_list[min_event_index]["task_id"] = task_id + 1
                    event_list[min_event_index]["start_time"] = time
                    working_time = self.actual_capability[officer_assigned][min_event_index, task_id + 1]
                    travelling_time = self.events.transition_matrix[location, min_event_index]
                    event_list[min_event_index]["end_time"] = time + working_time + travelling_time
                    event_list[min_event_index]["officer_assigned"] = officer_assigned
                    end_time[min_event_index] = event_list[min_event_index]["end_time"]
                    # Set Officer Occupied (Officer ID) = TRUE
                    officer_occupied[officer_assigned] = True
                    # Set Officer Location (Officer ID) = Event(EventID) Location
                    officer_location[officer_assigned] = min_event_index
                    # set event_task_status(EventId, Task number) = running
                    event_task_status[min_event_index, task_id + 1] = RUNNING
            # the officer is already occupied
            else:
                # Extract Officer ID assignment: Event(i)
                working_event = officer_location[officer_assigned]
                # If Event(EventID) has not yet started:
                if event_task_status[min_event_index, 0] == IDLE:
                    # set occurrence(Event(Event ID)) = Event(i).EndTime
                    occurrence[min_event_index] = event_list[working_event]["end_time"]
                # this event has been started
                else:
                    # Set Officer Occupied(Previous_officer ID) = FALSE
                    previous_officer = event_list[min_event_index]["officer_assigned"]
                    officer_occupied[previous_officer] = False
                    # set event_task_status(EventID, Task number) = Completed
                    task_id = event_list[min_event_index]["task_id"]
                    event_task_status[min_event_index, task_id] = COMPLETED
                    # set Event(EventID).End Time = Event(i).End Time
                    event_list[min_event_index]["end_time"] = event_list[working_event]["end_time"]
                    end_time[min_event_index] = event_list[min_event_index]["end_time"]
                    # set officer_assigned value in the dictionary to -1
                    # (This is to indicate that this event is waiting)
                    event_list[min_event_index]["officer_assigned"] = -2

        event_time_end = []
        for i in range(len(event_list)):
            event_time_end.append(event_list[i]["end_time"])

        check_end = np.sum(self.event_task_status_end - event_task_status)
        done = False if -1 in assignment else True
        assert check_end == 0  # make sure all events are completed

        event_time_completed = []
        for i in range(len(event_list)):
            event_time_completed.append(event_list[i]["end_time"] - event_list[i]["start_time"])

        self.event_time_end = event_time_end
        self.event_time_completed = event_time_completed
        self.obs = self.get_observation()
        reward = -np.max(event_time_end)

        return self.get_observation(), reward, done

    @staticmethod
    def legal_actions():
        # legals = [i for i, value in enumerate(self.action_table) if value == -1]
        # return legals
        legals = list(range(NUM_OFFICER))
        return legals

    def get_observation(self) -> np.ndarray:
        officers_capability = np.concatenate([capability.flatten() for capability in self.officers.capability])
        occurrence = self.events.occurrence
        transition_matrix = self.events.transition_matrix.flatten()
        action_table = self.action_table.flatten()
        obs = np.concatenate([officers_capability, occurrence, transition_matrix, action_table])
        return np.array([[obs]], dtype="int32")

    def render(self):
        print(self.obs)

    def get_length_obs(self):
        return len(self.obs[0][0])

    def action_to_matrix(self, action_number):
        allocation_index = self.action_table.tolist().index(-1)
        self.action_table[allocation_index] = action_number
        return self.action_table.reshape(self.events.num_event, self.events.num_task)

    @staticmethod
    def get_event_tie(min_event_indices, event_list, event_task_status):
        status = {}
        for index in min_event_indices:
            task_id = event_list[index]["task_id"]
            status[index] = event_task_status[index, task_id]

        # Extract EventID in priority of(Running > Idle > completed)
        for index in status:
            if status[index] == RUNNING:
                return index

        for index in status:
            if status[index] == IDLE:
                return index

        for index in status:
            if status[index] == COMPLETED:
                return index

        # safe-guard
        return None
