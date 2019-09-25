"""Simplest reinforcement learning model for finding the best path"""

import time
import pandas as pd
import numpy as np

np.random.seed(1)

"""Set parameters"""
N_STEPS = 6
EPSILON = 0.9 # GREEDY EPSILON
GAMMA = 0.9 # DECAY FACTOR
ACTIONS = ['left', 'right']
ALPHA = 0.1 # LEARNING RATE
MAX_EPISODE = 13
FRESH_TIME = 0.3 # FREQUENCY TO UPDATE A STEP


def q_table(nums_states: int, actions: list):
    table = pd.DataFrame(np.zeros([nums_states, len(actions)]), columns=actions)
    return table


def choose_action(state: int, table: pd.DataFrame):
    curr_state = table.iloc[state, :]
    if  np.random.uniform() > 0.9 or curr_state.any() == 0:
        return np.random.choice(table.columns)
    else:
        return curr_state.idxmax()


def env(state, action):
    if action == 'right':
        """reward right action"""
        if state == N_STEPS - 1:
            state_ = 'terminal'
            reward = 1
        else:
            state_ = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state_ = state
        else:
            state_ = state - 1
    return state_, reward


def print_env(episode, state, step):
    print_string = list('-' * N_STEPS)
    """print process"""
    if state != 'terminal':
        curr_string = print_string
        curr_string[state] = 'o'
        print('\r' + ''.join(curr_string), end='')
    else:
        interaction = 'Episode %d: total steps %d' % (episode, step)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                                ', end='')


if __name__ == '__main__':
    episode = 0
    table = q_table(N_STEPS, actions=ACTIONS)  # initialize the table
    while episode < MAX_EPISODE:
        state = 0
        step = 0
        while state != 'terminal':
            """require action and next state"""
            action = choose_action(state, table)
            state_, reward = env(state, action)
            step += 1
            """Update the table"""
            q_predict = table.loc[state, action]
            if state_ != 'terminal':
                q_target = reward + GAMMA * table.loc[state_, :].max()
            else:
                q_target = reward
            table.loc[state, action] = q_predict + ALPHA * (q_target - q_predict)
            state = state_
            print_env(episode, state, step)
            time.sleep(FRESH_TIME)

        """End of one episode"""
        episode += 1