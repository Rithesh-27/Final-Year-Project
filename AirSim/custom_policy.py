import torch
from torch import nn
from stable_baselines3.td3.policies import TD3Policy
from custom_networks import LidarFeatureExtractor, CustomActor, CustomCritic


class CustomDDPGPolicy(TD3Policy):

    def _build(self, lr_schedule):
        """
        Build custom feature extractor, actor, critic, and target nets.
        SB3 automatically calls this inside __init__.
        """
        # -----------------------
        # Feature extractors
        # -----------------------
        self.actor_fe = LidarFeatureExtractor().to(self.device)
        self.critic_fe = LidarFeatureExtractor().to(self.device)

        # -----------------------
        # Actor / Target Actor
        # -----------------------
        self.actor = CustomActor(self.actor_fe).to(self.device)
        self.actor_target = CustomActor(LidarFeatureExtractor().to(self.device)).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # -----------------------
        # Critic / Target Critic
        # -----------------------
        self.critic = CustomCritic(self.critic_fe).to(self.device)
        self.critic_target = CustomCritic(LidarFeatureExtractor().to(self.device)).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # -----------------------
        # Register parameters for SB3
        # -----------------------
        self._actor = self.actor
        self._critic = self.critic

        # Save shapes for SB3
        self.latent_dim = 48

    def _setup_model(self):
        """
        SB3 calls this automatically AFTER _build().
        This MUST define actor_optimizer and critic_optimizer.
        """

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor.optimizer = self.actor_optimizer

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.critic.optimizer = self.critic_optimizer

    # SB3 requires this for eval/train mode switching
    def set_training_mode(self, mode: bool) -> None:
        self.training = mode

        self.actor.train(mode)
        self.critic.train(mode)
        self.actor_target.train(mode)
        self.critic_target.train(mode)
