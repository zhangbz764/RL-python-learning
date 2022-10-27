import gym


class BespokeAgent:
    def __init__(self, env):
        pass

    # 决策
    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.06
        print(lb)
        print(ub)

        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action

    def learn(self, *args):
        pass


def play_montecarlo(env, agent, render=False, train=False):
    episode_rwd = 0.
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_rwd += reward

        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_rwd


if __name__ == '__main__':
    # select environment
    env = gym.make('MountainCar-v0')
    print('观测空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('动作数 = {}'.format(env.observation_space.shape))

    agent = BespokeAgent(env)
    episode_reward = play_montecarlo(env, agent, render=True)
    print('回合奖励 = {}'.format(episode_reward))
    env.close()
