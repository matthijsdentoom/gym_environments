from gym.envs.registration import register

register(
    id='tiling-pattern-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
)

register(
    id='tiling-pattern11x11-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
    kwargs={'x_dim': 11, 'y_dim': 11}
)

register(
    id='tiling-pattern7x5-static-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
    kwargs={'robots_path': 'robots.pickle', 'tiles_path': 'tiles.npy'}
)

register(
    id='tiling-pattern11x11-static-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
    kwargs={'robots_path': 'robots.pickle', 'tiles_path': 'tiles11x11.npy'}
)

# This environment has all the tiles distributed in a 6x6 block giving a fitness of 67.24
register(
    id='tiling-pattern11x11-block-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
    kwargs={'robots_path': 'robots.pickle', 'tiles_path': 'tiles11x11_block.npy'}
)
