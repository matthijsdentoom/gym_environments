import gym
import gym_multi_robot


if __name__ == '__main__':
    env = gym.make('tiling-pattern-v0')

    while True:
        env.render()
