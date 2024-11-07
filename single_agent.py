from metadrive import MetaDriveEnv
import tqdm
import gym
import numpy as np
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

training_env = MetaDriveEnv(dict(
    num_scenarios=1000,
    start_seed=1000,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True
))


test_env = MetaDriveEnv(dict(
    num_scenarios=200,
    start_seed=0,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True
))

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("meta-drive")
