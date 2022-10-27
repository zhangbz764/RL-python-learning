import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible

# constants
N_STATES = 10  # the length of the 1D world
ACTIONS = ['left', 'right']  # avilable actions
EPSILON = 0.9  # ε-greedy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODE = 50
FRESH_TIME = 0.2


def init_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    print(table)
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(state, action):
    if action == 'right':
        # move right
        if state == N_STATES - 2:
            state_next = 'terminal'
            reward = 1
        else:
            state_next = state + 1
            reward = 0
    else:
        # move left
        reward = 0
        if state == 0:  # left end
            state_next = state
        else:
            state_next = state - 1
    return state_next, reward


def update_env(S, episode, step_counter):
    env_list = ['_'] * (N_STATES - 1) + ['H']
    if S == 'terminal':
        interation = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interation), end='')
        time.sleep(2)
        print('\r')
    else:
        env_list[S] = 'o'
        interation = ''.join(env_list)
        print('\r{}'.format(interation), end='')
        time.sleep(FRESH_TIME)


def perform_rl():
    q_table = init_q_table(N_STATES, ACTIONS)
    for eps in range(MAX_EPISODE):
        # every learning episode
        step_counter = 0
        S = 0
        terminated = False

        update_env(S, eps, step_counter)
        while not terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]  # 估计值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # 真实值
            else:
                q_target = R
                terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update table
            S = S_  # next state

            update_env(S, eps, step_counter)
            step_counter += 1

        print('\n' + 'Q-TABLE: ')
        print(q_table)
    return q_table


if __name__ == '__main__':
    q_table = perform_rl()

