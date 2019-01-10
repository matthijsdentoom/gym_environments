"""
Simple example using the tile structure creation task.
"""

from __future__ import print_function
import os
import neat
import gym
import gym_multi_robot

num_steps = 3000
num_robots = 5
num_trials = 10

env = gym.make('tiling-pattern-v0')


def eval_genomes(genomes, config):
    count = 0
    for genome_id, genome in genomes:
        print(count)
        count += 1
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = run_environment(net)


def run_environment(net):
    reward = 0
    for _ in range(num_trials):
        observation = env.reset()

        sub_reward = 0

        for i in range(num_steps):
            output = [net.activate(observation[i]) for i in range(len(observation))]
            observation, sub_reward, done, info = env.step(output)

        reward += sub_reward

    return reward / num_trials


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)