from stable_baselines3 import DDPG, PPO
from airsim_drone_env import AirSimDroneDDPGEnv
from custom_policy import CustomDDPGPolicy, SimpleDDPGPolicy
import torch
from blocks_script import start_blcoks, stop_blocks
from custom_networks import CustomActor, LidarFeatureExtractor
import numpy as np

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

checkpoint_callback = CheckpointCallback(
    save_freq=1000,      # timesteps
    save_path="./checkpoints",
    name_prefix="ddpg_airsim"
)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # env is vectorized → index 0
        env = self.training_env.envs[0].unwrapped

        # Safely log attributes from env
        self.logger.record("nav/distance", env.get_distance_to_goal())

        if hasattr(env, "last_reward"):
            self.logger.record("reward/total", env.last_reward)

        if self.n_calls % 100 == 0:
            self.logger.dump(step=self.num_timesteps)

        return True

tensorboard_callback = TensorboardCallback()

start_blcoks()

env = AirSimDroneDDPGEnv()
# actor = CustomActor(LidarFeatureExtractor())
# obs, _ = env.reset()
# obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


# for i in range(10):
#     action = actor(obs)
#     action = action.detach().numpy()[0]
#     print(f"Action given: {action}\n")
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated:
#         print("\n\nTermintaed\n\n")
#         break
#     obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#     print(f"Reward obtained: {reward}\n\n")

n_actions = env.action_space.shape[0]

action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.3 * np.ones(n_actions),   # start here
    theta=0.15
)

model = DDPG(
    policy=CustomDDPGPolicy,
    env=env,
    verbose=1,
    batch_size=32,
    buffer_size=10000,
    learning_rate=1e-3,
    action_noise=action_noise,
    tensorboard_log="./tb_logs/",
    gamma=0.7
)

model.policy._setup_model()

model.learn(10000, callback=[checkpoint_callback])
model.save("ddpg_model")

stop_blocks()