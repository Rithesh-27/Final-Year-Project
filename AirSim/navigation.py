import time
import numpy as np
import airsim

from stable_baselines3 import DDPG
from airsim_drone_env import AirSimDroneDDPGEnv
from blocks_script import start_blcoks, stop_blocks
import torch
from custom_policy import CustomDDPGPolicy

start_blcoks()

MODEL_PATH = "checkpoints/ddpg_airsim_1000_steps.zip"
MAX_STEPS = 2000
SLEEP_TIME = 0.05

env = AirSimDroneDDPGEnv(inference_mode=True)

model = DDPG(
    policy=CustomDDPGPolicy,
    env=env,
    verbose=1,
    batch_size=32,
    buffer_size=10000,
    learning_rate=1e-3,
    tensorboard_log="./tb_logs/"
)

# Setup the model structure
model.policy._setup_model()

# Load the saved parameters
model.set_parameters(MODEL_PATH)

obs, _ = env.reset()

print("Starting navigation...\n")

for step in range(MAX_STEPS):

    action, _ = model.predict(obs, deterministic=True)
    # print(f"Action received: {action}")

    obs, reward, terminated, truncated, _ = env.step(action)

    time.sleep(SLEEP_TIME)

    if terminated or truncated:
        print(f"Episode finished at step {step}")
        break

print("Navigation finished")

stop_blocks()

