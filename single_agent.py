from metadrive import MetaDriveEnv
import tqdm
import gymnasium as gym
import numpy as np
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from metadrive.envs.gym_wrapper import createGymWrapper

training_env = MetaDriveEnv(dict(
    num_scenarios=1000,
    start_seed=1000,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True
))
# gym_training_env = createGymWrapper(MetaDriveEnv)(env_config)
gym_training_env = createGymWrapper(training_env)

model = SAC(MlpPolicy, gym_training_env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("meta-drive")
