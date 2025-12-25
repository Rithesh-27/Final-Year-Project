import gymnasium as gym
import numpy as np
from gymnasium import spaces
import airsim
import time
import math
from set_course import build_course

class AirSimDroneDDPGEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------
    def __init__(self, inference_mode = False):

        super(AirSimDroneDDPGEnv, self).__init__()

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.inference_mode = inference_mode

        # Sandbox area coords
        self.SX = 500
        self.SY = 500
        self.SZ = -1.5

        build_course(self.client)

        # Goal position relative to start
        self.goals = [
            # ---- start → wall set 1 ----
            np.array([1,   1.5, 0]),
            np.array([2,   1.5, 0]),
            np.array([4,   2.5, 0]),
            np.array([5,   3.0, 0]),

            # ---- wall set 1 → gap ----
            np.array([7,   2.0, 0]),
            np.array([9,   1.0, 0]),
            np.array([11,  0.0, 0]),

            # ---- gap → left turn ----
            np.array([13,  1.5, 0]),
            np.array([15,  3.0, 0]),
            np.array([17,  3.0, 0]),

            # ---- left turn → zig 1 ----
            np.array([19,  2.0, 0]),
            np.array([21, -1.0, 0]),
            np.array([24, -3.0, 0]),

            # ---- zig 1 → zig 2 ----
            np.array([25, -1.0, 0]),
            np.array([26,  1.0, 0]),
            np.array([27,  2.0, 0]),

            # ---- zig 2 → zig 3 ----
            np.array([29,  0.0, 0]),
            np.array([30, -2.0, 0]),
            np.array([31, -3.0, 0]),

            # ---- zig 3 → goal ----
            np.array([34, -2.0, 0]),
            np.array([38, -1.0, 0]),
            np.array([42,  0.0, 0]),
        ]

        
        self.current_idx = 0
        self.prev_distance = None
        self.episode_reward = 0

        # ---------------------------
        # ACTION SPACE: [vx, yaw_rate]
        # ---------------------------
        self.action_space = spaces.Box(
            low=np.array([-1, -1.5], dtype=np.float32),
            high=np.array([1, 1.5], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.H = 64

        # ---------------------------
        # OBSERVATION SPACE
        # lidar (4,H,W) + 2 scalars
        # we assume lidar depth image = (64×256)
        # ---------------------------
        self.frame_stack = 4

        self.observation_space = spaces.Box(
            low  = -1.0,
            high = 1.0,
            shape=(4,64,258),
            dtype=np.float32
        )

        # For reward tracking
        self.prev_distance = None
        self.max_steps = 350
        self.steps = 0

    # ------------------------------------------------------------------
    # RESET EPISODE
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.client.hoverAsync().join()

        self.steps = 0
        self.episode_reward = 0
        self.current_idx = 0
        # Reset drone
        pose = airsim.Pose(
            airsim.Vector3r(self.SX, self.SY, self.SZ),
            airsim.to_quaternion(0, 0, 0)
        )
        self.client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.5)

        # Initialize lidar stack buffer
        lidar = self.get_lidar()
        self.lidar_stack = np.stack([lidar]*self.frame_stack, axis=0)

        # Initial distance
        self.prev_distance = self.get_distance_to_goal()

        # Observation includes lidar stack + (d, θ)
        obs = self.build_observation()

        return obs, {}

    # ------------------------------------------------------------------
    # STEP FUNCTION
    # ------------------------------------------------------------------
    def step(self, action):

        self.steps += 1

        # ---- Apply action ----
        vx = float(action[0] * 1.5)
        yaw_rate = float(action[1] * 30)

        # Get current yaw
        pose = self.client.simGetVehiclePose()
        current_yaw = airsim.to_eularian_angles(pose.orientation)[2]
        
        # Transform to world frame
        vx_world = vx * np.cos(current_yaw)
        vy_world = vx * np.sin(current_yaw)

        self.prev_distance = self.get_distance_to_goal()
        
        # Execute command
        self.client.moveByVelocityZAsync(
            vx=vx_world,
            vy=vy_world,
            z=self.SZ,
            duration=0.15,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        ).join()

        # ---- Collect new state ----
        lidar = self.get_lidar()
        self.lidar_stack = np.roll(self.lidar_stack, shift=-1, axis=0)
        self.lidar_stack[-1] = lidar

        distance = self.get_distance_to_goal()
        terminated, truncated, crash, success = self.check_termination()

        reward = self.get_reward(np.array([vx,yaw_rate]), distance, crash, success)
        pos = self.client.simGetVehiclePose().position
        drone = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=float)

        obs = self.build_observation()
        self.episode_reward += reward

        if terminated or truncated:
            self.last_reward = self.episode_reward

        return obs, reward, terminated, truncated, {}
    
    def get_reward(self, action, distance, crash, success):
        progress = self.prev_distance - distance
        distance_reward = 0.0

        if distance < 1.5:
            distance_reward += 25.0 * (self.current_idx + 1)

        distance_penalty = -0.015 * (distance ** 2)

        bearing = self.get_bearing()
        bearing_penalty = -0.75 * abs(bearing)
        # -------------------------------------------------
        # 3. Small action regularization
        # -------------------------------------------------
        yaw_penalty = -0.002 * abs(action[1])
        stagnation_penalty = -0.1 if abs(action[0]) < 0.15 else 0.0

        # -------------------------------------------------
        # Total reward
        # -------------------------------------------------
        reward = (
            distance_reward +
            distance_penalty + 
            bearing_penalty +
            yaw_penalty +
            stagnation_penalty
        )

        # -------------------------------------------------
        # 4. Terminal rewards
        # -------------------------------------------------
        if success:
            reward += 500.0

        if crash:
            reward -= 30.0

        return float(reward)
    
    # ------------------------------------------------------------------
    # OBSERVATION CONSTRUCTION
    # ------------------------------------------------------------------
    def build_observation(self):
        # lidar = (4, H, W)
        # append distance + theta as 2 extra columns

        d = self.get_distance_to_goal()
        theta = self.get_bearing()

        max_dist = 50.0
        max_bearing = np.pi
        # Broadcast d and theta to column vectors
        d_col = np.ones((self.frame_stack, self.H, 1), dtype=np.float32) * d
        t_col = np.ones((self.frame_stack, self.H, 1), dtype=np.float32) * theta

        d_col = d_col / max_dist
        t_col = t_col / max_bearing

        obs = np.concatenate([self.lidar_stack, d_col, t_col], axis=2, dtype=np.float32)
        return obs


    # ------------------------------------------------------------------
    # LIDAR PROCESSING: Convert to depth image
    # ------------------------------------------------------------------
    def get_lidar(self):
        # Flushing previous lidar images
        _ = self.client.getLidarData(lidar_name="Lidar1", vehicle_name="Drone1")
        time.sleep(0.05)
        
        lidar = self.client.getLidarData(lidar_name="Lidar1", vehicle_name="Drone1")
        if len(lidar.point_cloud) < 3:
            return np.zeros((self.H, self.W), dtype=np.float32)

        pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1,3)

        def lidar_to_depth_image(pts,num_channels=64,h_bins=256,v_fov_up=15.0,v_fov_down=-15.0,max_range=50.0):
            """
            Convert LiDAR point cloud to cylindrical depth image.

            pts: (N,3) point cloud in sensor frame
            returns: (num_channels, h_bins) depth image
            """

            if pts.shape[0] == 0:
                return np.ones((num_channels, h_bins), dtype=np.float32)

            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

            # -------------------------
            # 1. Compute range
            # -------------------------
            r = np.sqrt(x**2 + y**2 + z**2)
            valid = (r > 0.1) & (r < max_range)
            x, y, z, r = x[valid], y[valid], z[valid], r[valid]

            # -------------------------
            # 2. Azimuth (horizontal)
            # -------------------------
            azimuth = np.arctan2(y, x)           # [-pi, pi]
            azimuth = (azimuth + np.pi) / (2 * np.pi)  # [0,1]
            col = (azimuth * h_bins).astype(np.int32)
            col = np.clip(col, 0, h_bins - 1)

            # -------------------------
            # 3. Elevation (vertical)
            # -------------------------
            elevation = np.arcsin(z / r) * 180 / np.pi  # degrees

            v_range = v_fov_up - v_fov_down
            row = (1.0 - (elevation - v_fov_down) / v_range) * num_channels
            row = row.astype(np.int32)
            row = np.clip(row, 0, num_channels - 1)

            # -------------------------
            # 4. Create depth image
            # -------------------------
            depth = np.ones((num_channels, h_bins), dtype=np.float32) * max_range

            for i in range(r.shape[0]):
                if r[i] < depth[row[i], col[i]]:
                    depth[row[i], col[i]] = r[i]

            # -------------------------
            # 5. Normalize
            # -------------------------
            depth = depth / max_range
            depth = np.clip(depth, 0.0, 1.0)

            return depth


        # Project to 2D depth image (simple approach)
        depth_img = lidar_to_depth_image(pts)

        return depth_img.astype(np.float32)


    # ------------------------------------------------------------------
    # DISTANCE + BEARING TO GOAL
    # ------------------------------------------------------------------
    def get_distance_to_goal(self):
        pos = self.client.simGetVehiclePose().position
        drone = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=float)
        goal = np.array([self.SX + self.goals[self.current_idx][0], self.SY + self.goals[self.current_idx][1], self.SZ + self.goals[self.current_idx][2]])
        dist = np.linalg.norm(drone - goal)
        return dist

    def get_bearing(self):
        pose = self.client.simGetVehiclePose()
        pos = pose.position
        yaw = airsim.to_eularian_angles(pose.orientation)[2]

        dx = (self.SX + self.goals[self.current_idx][0]) - pos.x_val
        dy = (self.SY + self.goals[self.current_idx][1]) - pos.y_val

        goal_angle = math.atan2(dy, dx)
        bearing = goal_angle - yaw

        # wrap to [-pi, pi]
        return float((bearing + np.pi) % (2 * np.pi) - np.pi)
    
    # ------------------------------------------------------------------
    # TERMINATION CHECK
    # ------------------------------------------------------------------
    def check_termination(self):

        # Check collision
        collision = self.client.simGetCollisionInfo()
        if collision.has_collided and "Ground" not in collision.object_name :
            print(f"Collided with {collision.object_name}\n")
            return True, False, True, False

        # Goal reached
        if self.get_distance_to_goal() < 1.5:
            print(f"Goal {self.current_idx} reached\n")
            self.current_idx += 1
            if self.current_idx == len(self.goals):
                return True, False, False, True
            self.prev_distance = self.get_distance_to_goal()
            return False, False, False, False
        
        # Out of bounds check
        pos = self.client.simGetVehiclePose().position
        x, y, z = pos.x_val, pos.y_val, pos.z_val

        # X bounds (maze corridor from x=0 to x=42)
        if x < self.SX - 2 or x > self.SX + 45:
            print("X out of bounds\n")
            return True, False, False, False

        # Y bounds (walls at y = ±6 → give margin to ±7)
        if y < self.SY - 6 or y > self.SY + 6:
            print("Y out of bounds\n")
            return True, False, False, False

        # Z bounds (walls are about ~2m high)
        if z < -2.5 or z > 1.0 :  # center-based limits
            print("Z out of bounds\n")
            return True, False, False, False
        
        # Timeout
        if self.steps >= self.max_steps:
            print("Timeout\n")
            return False, True, False, False

        return False, False, False, False


    def render(self, mode="human"):
        pass

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
