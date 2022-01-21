import numpy as np


class EpsGreedy:
    def __init__(self, epsilon, eps_decay, officers, events):
        super().__init__()
        # Set number of arms
        self.narms = 0
        # Exploration probability
        self.epsilon = epsilon
        # Decay rate
        self.eps_decay = eps_decay
        # Total step count
        self.step_n = 0

        # officers and events
        self.officers = officers
        self.events = events

    # Play one round and return the action (chosen arm)
    def play(self, tround, context=None):
        # Generate random number
        p = np.random.rand()

        # Edge case, epsilon = 0, and the first step, we still random
        if self.epsilon == 0 and self.step_n == 0:
            action = self.random_action()

        # This case, it goes exploration
        elif p < self.epsilon:
            action = self.random_action()
        # Here, it goes exploitation
        else:
            arm = self.AM_reward.argMax()
            action = np.fromstring(arm, dtype=int).reshape((self.events.num_event, self.events.num_task))

        self.tround = tround
        # get the chosen arm
        self.arm = action.tostring()
        self.action = action

        self.epsilon = max(0, self.epsilon - self.eps_decay)

        # print("Action chosen: \n{}".format(self.action))
        return action

    def random_action(self):
        action = np.random.randint(0, self.officers.num_officer,
                                   size=(self.events.num_event, self.events.num_task))
        return action

    def update(self, action, reward, context=None):
        # get the context (may be None)
        self.context = context
        # update the overall step of the model
        self.step_n += 1
        # update the step of individual arms
        self.step_arm[self.arm] += 1
        # update average mean reward of each arm
        self.AM_reward[self.arm] = ((self.step_arm[self.arm] - 1)
                                    / float(self.step_arm[self.arm])
                                    * self.AM_reward[self.arm]
                                    + (1 / float(self.step_arm[self.arm])) * reward)
