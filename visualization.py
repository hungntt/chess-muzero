from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


class Visualization:
    def __init__(self, game, muzero=None, random=None):
        self.game = game
        self.muzero = muzero
        self.random = random

    def visualize(self):
        df = pd.DataFrame(data={"random": self.random, "muzero": self.muzero})
        plt.figure()
        df["avg_random"] = df.loc[:, "random"].rolling(window=100).mean()
        df["avg_muzero"] = df.loc[:, "muzero"].rolling(window=100).mean()
        df[["avg_random", "avg_muzero"]].plot()
        plt.suptitle(f'MuZero vs Random')
        plt.title("Performance comparison Random vs DQN")
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        # plt.show()
        plt.savefig(f'{self.game.arena_path}/MuZero_vs_random.png')

