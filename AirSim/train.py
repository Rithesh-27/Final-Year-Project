from stable_baselines3 import DDPG
from airsim_drone_env import AirSimDroneDDPGEnv
from custom_policy import CustomDDPGPolicy
import torch

env = AirSimDroneDDPGEnv()

model = DDPG(
    policy=CustomDDPGPolicy,
    env=env,
    verbose=1,
    batch_size=64,
    buffer_size=10000,
    learning_rate=1e-3,
    gamma=0.99,
    tau=0.005
)

model.policy._setup_model()

model.learn(1_000_000)
model.save("ddpg_model")