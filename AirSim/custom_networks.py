import torch
import torch.nn as nn
import torch.nn.functional as F

class LidarFeatureExtractor(nn.Module):
    """
    Correct CNN + GRU feature extractor.
    Processes ONLY depth images, NOT distance/theta.
    
    Input:  obs = (batch, 4, 64, 256+2)
    We slice: depth = obs[:, :, :, :256]
    """
    def __init__(self):
        super().__init__()

        # CNN layers (applied time-distributed)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)

        # After conv layers:
        # Output size = 16 * 64 * 256 = 262,144 features per frame
        self.flatten_size = 32 * 8 * 32

        # GRU over 4 time steps
        self.gru = nn.GRU(
            input_size=self.flatten_size,
            hidden_size=48,
            batch_first=True
        )

    def forward(self, obs):

        # ---------- 1. Keep only the depth image ----------
        # obs shape: (batch, 4, 64, 258)
        depth = obs[:, :, :, :256]          # -> (batch, 4, 64, 256)

        # ---------- 2. Add channel dim for CNN ----------
        depth = depth.unsqueeze(2)          # -> (batch, 4, 1, 64, 256)

        batch = depth.shape[0]

        # ---------- 3. Flatten time dimension and apply CNN ----------
        depth = depth.reshape(batch * 4, 1, 64, 256)
        x = F.relu(self.conv1(depth))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # ---------- 4. Flatten CNN output ----------
        x = x.reshape(batch * 4, self.flatten_size)

        # ---------- 5. Reshape back into sequence ----------
        x = x.reshape(batch, 4, self.flatten_size)

        # ---------- 6. GRU across time ----------
        gru_out, _ = self.gru(x)
        x = gru_out[:, -1, :]    # last time step output

        return x     # shape: (batch, 48)
    
    
class CustomActor(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.fe = feature_extractor

        # 48 + 2 (d, theta)
        self.fc1 = nn.Linear(48 + 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64,2)

    def forward(self, obs):
        extras = obs[:,0,0,256:]

        x = self.fe(obs)
        x = torch.cat([x,extras], dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.tanh(self.fc3(x))
        return x
    
class CustomCritic(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.fe = feature_extractor

        # Critic 1
        self.fc1_1 = nn.Linear(48 + 2 + 2, 64)
        self.fc2_1 = nn.Linear(64, 64)
        self.fc3_1 = nn.Linear(64, 1)

        # Critic 2 (independent parameters)
        self.fc1_2 = nn.Linear(48 + 2 + 2, 64)
        self.fc2_2 = nn.Linear(64, 64)
        self.fc3_2 = nn.Linear(64, 1)

    def q1_forward(self, obs, action):
        """Return only the first Q-value (SB3 uses this for actor loss)."""
        extras = obs[:, 0, 0, 256:]
        x = self.fe(obs)
        x = torch.cat([x, extras, action], dim=1)

        q1 = torch.relu(self.fc1_1(x))
        q1 = torch.relu(self.fc2_1(q1))
        q1 = self.fc3_1(q1)
        return q1

    def forward(self, obs, action):
        extras = obs[:, 0, 0, 256:]
        x = self.fe(obs)
        x = torch.cat([x, extras, action], dim=1)

        # ---------------- Q1 ----------------
        q1 = torch.relu(self.fc1_1(x))
        q1 = torch.relu(self.fc2_1(q1))
        q1 = self.fc3_1(q1)

        # ---------------- Q2 ----------------
        q2 = torch.relu(self.fc1_2(x))
        q2 = torch.relu(self.fc2_2(q2))
        q2 = self.fc3_2(q2)

        return q1, q2

class SimpleActor(nn.Module):
    def __init__(self, obs_dim=2, act_dim=2):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)

    def forward(self, obs):
        """
        obs: (batch, 2) -> [distance, bearing]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # [-1, 1]
        return action

class SimpleCritic(nn.Module):
    def __init__(self, obs_dim=2, act_dim=2):
        super().__init__()

        # Q1 network
        self.q1_fc1 = nn.Linear(obs_dim + act_dim, 64)
        self.q1_fc2 = nn.Linear(64, 64)
        self.q1_fc3 = nn.Linear(64, 1)

        # Q2 network
        self.q2_fc1 = nn.Linear(obs_dim + act_dim, 64)
        self.q2_fc2 = nn.Linear(64, 64)
        self.q2_fc3 = nn.Linear(64, 1)

    def q1_forward(self, obs, action):
        """
        Used by SB3 for actor loss
        """
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.q1_fc1(x))
        x = F.relu(self.q1_fc2(x))
        q1 = self.q1_fc3(x)
        return q1

    def forward(self, obs, action):
        """
        Returns Q1, Q2
        """
        xu = torch.cat([obs, action], dim=1)

        # Q1
        q1 = F.relu(self.q1_fc1(xu))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        # Q2
        q2 = F.relu(self.q2_fc1(xu))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2
