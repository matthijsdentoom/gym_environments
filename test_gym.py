import gym
import gym_multi_robot


if __name__ == '__main__':
    env = gym.make('tiling-pattern-v0')
    env.render()

    input('Turn right')
    env.step([[False, 1, False, False] for _ in range(5)])
    env.render()
    #
    # input('Turn left')
    # env.step([[False, -1, False, False] for _ in range(5)])
    # env.render()

    # input('press key to pickup')
    # env.step([[False, 0, True, False] for _ in range(5)])
    # env.render()
    #
    # input('press key to dropdown')
    # env.step([[False, 0, False, True] for _ in range(5)])
    # env.render()

    # input('Press key to get up')
    # env.step([[True, 0, False, False] for _ in range(5)])
    # env.render()

    input('Press key to exit')
