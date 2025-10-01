"""
changelog:
24.9.25  optionally changed definition of best from longest survival to longest survival + best mean reward 
         if same length (at a certain point, many runs will survive the whole period) 
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.gridspec as gridspec
import pickle 

def sliding_mean(arr, window_size=100):
    kernel = np.ones(window_size) / window_size
    means = np.convolve(arr, kernel, mode='valid')
    # Pad with zeros at the beginning to match original length
    padded = np.concatenate([np.zeros(window_size - 1), means])
    return padded

class Logger:
    
    def __init__(self, data_labels, n_avg = 20, n_update_data = 1, dump_every=50, name=None):
        self.n_avg = n_avg
        self.n_update_data = n_update_data
        self.name = name
        self.dump_every = dump_every
        self.data_labels = data_labels

        self.episode_durations = []
        self.best_data = None
        self.best_steps = 0
        self.best_mean_reward = -np.inf
        self.idx_best_run = None
        self.data = []

        plt.ion()

        # Set up the figure and grid layout
        n_rows = len(data_labels)
        y_subfigsize = 3
        fig = plt.figure(figsize=(12, n_rows*y_subfigsize))
        gs = gridspec.GridSpec(n_rows, 3, width_ratios=[2, 2, 2])
        # Full height single column
        self.left_ax = fig.add_subplot(gs[:, 0])  

        # grid axes
        self.axes = []
        for col in range(2):
            row_axes = []
            for row,_ in enumerate(data_labels):
                ax = fig.add_subplot(gs[row, col+1])
                row_axes.append(ax)
            self.axes.append(row_axes)

    def push_show(self, data, final_step = False, this_rewards = None):
        """
        live plots: every element in data - data is a list of 1d iterables
                    steps (length of data elements) over time incl mean
                    best data - according to steps and optionally reward
                    steps is updated every call, data only n_update_data times 
        """
        steps = min([len(di) for di in data])
        this_idx = len(self.episode_durations)
        this_mean_reward = np.mean(this_rewards) if this_rewards is not None else -np.inf
        self.episode_durations.append(steps)
        self.data.append(data)

        if len(data) % self.dump_every == 0 or final_step:
            self.dump()

        new_best = False
        if steps > self.best_steps or (steps == self.best_steps and this_mean_reward > self.best_mean_reward):
            self.best_steps = steps
            self.best_data = data
            self.idx_best_run = this_idx
            self.best_mean_reward = this_mean_reward
            new_best = True

        
        durations = np.array(self.episode_durations)
        self.left_ax.clear()
        self.left_ax.plot(durations, label="current")
        # Take n episode averages and plot them too
        if len(durations) >= self.n_avg:
            means = sliding_mean(durations, self.n_avg)
            self.left_ax.plot(means, label=f"sliding window(-{self.n_avg}) mean")
        self.left_ax.set_xlabel('run')
        self.left_ax.set_ylabel('steps')
        self.left_ax.set_title("steps per run")
        self.left_ax.legend()

        for col,(col_data,col_label) in enumerate([(data,           f"current: {this_idx}"),
                                                   (self.best_data, f"best run: {self.idx_best_run}")]):
            #  update current every so often                    update best when available
            if col+1==1 and this_idx % self.n_update_data == 0 or col+1==2 and new_best:
                for row,(di,row_label) in enumerate(zip(col_data, self.data_labels)):
                    self.axes[col][row].clear()
                    self.axes[col][row].plot(di)
                    self.axes[col][row].set_title(row_label + ", " + col_label)

        # plt.figure(1)
        plt.tight_layout()

        plt.pause(0.01)  # pause a bit so that plots are updated

        if 'inline' in matplotlib.get_backend():
            from IPython import display
            if not final_step:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

        if final_step:
            plt.ioff()
            plt.show()

    def dump(self):
        if self.name is None:
            return
        
        fname = self.name + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self.data, f)

class Train_Logger:
    
    def __init__(self, data_labels, n_update_data = 1, dump_every=50, name=None):

        self.n_update_data = n_update_data
        self.name = name
        self.dump_every = dump_every
        self.data_labels = data_labels

        self.data = [[] for _ in data_labels]

        plt.ion()
        # Set up the figure and grid layout
        n_rows = len(data_labels)
        y_subfigsize = 1.5
        fig = plt.figure(figsize=(7, n_rows*y_subfigsize))
        gs = gridspec.GridSpec(n_rows,1, width_ratios=[2])

        # grid axes
        self.axes = []
        col = 0
        row_axes = []
        for row,_ in enumerate(data_labels):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        self.axes.append(row_axes)

    def push_show(self, data=None, final_step = False):
        
        this_idx = len(self.data[0])

        if data is not None:
            for i,di in enumerate(data):
                self.data[i].append(di)

        if len(self.data[0]) % self.dump_every == 0 or final_step:
            self.dump()

        col,col_data = 0, self.data
        col_label = "finished" if final_step else f"current: {this_idx}"
        #  update current every so often                   
        if col+1==1 and this_idx % self.n_update_data == 0:
            for row,(di,row_label) in enumerate(zip(col_data, self.data_labels)):
                self.axes[col][row].clear()
                self.axes[col][row].plot(di)
                self.axes[col][row].set_title(row_label + ", " + col_label)

        # plt.figure(1)
        plt.tight_layout()

        plt.pause(0.01)  # pause a bit so that plots are updated

        if 'inline' in matplotlib.get_backend():
            from IPython import display
            if not final_step:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

        if final_step:
            plt.ioff()
            plt.show()


    def dump(self):
        if self.name is None:
            return

        fname = self.name + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self.data, f)