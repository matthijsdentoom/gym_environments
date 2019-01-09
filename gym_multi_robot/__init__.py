from gym.envs.registration import register

register(
    id='tiling-pattern-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
)
