import gymnasium as gym
import numpy as np
from gymnasium import spaces
import airsim
import cv2
import time
import math
from set_course import build_course
import cv2

class AirSimDroneDDPGEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------
    def __init__(self):

        super(AirSimDroneDDPGEnv, self).__init__()

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Sandbox area coords
        self.SX = 500
        self.SY = 500
        self.SZ = -5
        self.GZ = -2
        self.OBZ = 1

        build_course(self.client)

        # Goal position relative to start
        self.goal = np.array([35, 0, 0], dtype=np.float32)

        # ---------------------------
        # ACTION SPACE: [vx, yaw_rate]
        # ---------------------------
        self.action_space = spaces.Box(
            low=np.array([-1.5, -1.5], dtype=np.float32),
            high=np.array([1.5, 1.5], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # ---------------------------
        # OBSERVATION SPACE
        # lidar (4,H,W) + 2 scalars
        # we assume lidar depth image = (64×256)
        # ---------------------------
        self.H = 64
        self.W = 256
        self.frame_stack = 4

        low = -np.inf
        high = np.inf

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.frame_stack, self.H, self.W + 2),
            dtype=np.float32
        )

        # For reward tracking
        self.prev_distance = None
        self.max_steps = 1000
        self.steps = 0

    # ------------------------------------------------------------------
    # RESET EPISODE
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.steps = 0

        # Reset drone
        pose = airsim.Pose(
            airsim.Vector3r(self.SX, self.SY, 0),
            airsim.to_quaternion(0, 0, 0)
        )
        self.client.simSetVehiclePose(pose, ignore_collision=True)

        # Initialize lidar stack buffer
        lidar = self.get_lidar()
        self.lidar_stack = np.stack([lidar]*self.frame_stack, axis=0)

        # Initial distance
        self.prev_distance = self.get_distance_to_goal()

        # Observation includes lidar stack + (d, θ)
        obs = self.build_observation()

        return obs, {}

    def get_heading_alignment(self):
        """
        Returns cos(theta) where:
        - 1.0 means drone facing toward goal
        - -1.0 means facing away
        """
        drone_state = self.client.getMultirotorState().kinematics_estimated
        drone_yaw = airsim.to_eularian_angles(drone_state.orientation)[2]

        dx, dy = self.goal[0] - drone_state.position.x_val, self.goal[1] - drone_state.position.y_val
        goal_angle = np.arctan2(dy, dx)

        return np.cos(goal_angle - drone_yaw)


    # ------------------------------------------------------------------
    # STEP FUNCTION
    # ------------------------------------------------------------------
    def step(self, action):

        self.steps += 1

        # ---- Apply action ----
        vx = float(action[0]) * 10
        yaw_rate = float(action[1] * 30)

        self.client.moveByVelocityAsync(
            vx, 0, 0,
            duration=0.1,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )
        time.sleep(0.1)

        # ---- Collect new state ----
        lidar = self.get_lidar()
        self.lidar_stack = np.roll(self.lidar_stack, shift=-1, axis=0)
        self.lidar_stack[-1] = lidar

        distance = self.get_distance_to_goal()
        terminated, truncated, crash, success = self.check_termination()

        # ---- Reward Calculation ----
        progress = self.prev_distance - distance       # + if drone is moving toward goal
        heading = self.get_heading_alignment()         # [-1,1], 1 means facing goal

        # Step penalty to prevent stagnation
        step_penalty = -0.01

        # Reward for moving toward goal
        progress_reward = progress * 5.0               # amplify signal

        # Heading reward
        heading_reward = heading * 0.5                 # drone should face goal

        # Velocity reward (encourage movement)
        speed_reward = abs(vx) * 0.02

        # Negative reward for small velocities (hovering)
        stagnation_penalty = 0
        if abs(vx) < 0.2:
            stagnation_penalty = -0.05

        # Combine
        reward = progress_reward + heading_reward + speed_reward + step_penalty + stagnation_penalty

        # Terminal events
        if success:
            reward = 20          # much higher goal reward
        if crash:
            reward = -10         # stronger crash penalty

        obs = self.build_observation()

        return obs, reward, terminated, truncated, {}


    # ------------------------------------------------------------------
    # OBSERVATION CONSTRUCTION
    # ------------------------------------------------------------------
    def build_observation(self):
        # lidar = (4, H, W)
        # append distance + theta as 2 extra columns

        d = self.get_distance_to_goal()
        theta = self.get_bearing()

        # Broadcast d and theta to column vectors
        d_col = np.ones((self.frame_stack, self.H, 1), dtype=np.float32) * d
        t_col = np.ones((self.frame_stack, self.H, 1), dtype=np.float32) * theta

        obs = np.concatenate([self.lidar_stack, d_col, t_col], axis=2)
        return obs.astype(np.float32)


    # ------------------------------------------------------------------
    # LIDAR PROCESSING: Convert to depth image
    # ------------------------------------------------------------------
    def get_lidar(self):
        lidar = self.client.getLidarData(lidar_name="Lidar1", vehicle_name="Drone1")
        if len(lidar.point_cloud) < 3:
            return np.zeros((self.H, self.W), dtype=np.float32)

        pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1,3)

        def lidar_to_depth_image(points, img_h=64, img_w=256, max_range=100.0):
            """
            Convert AirSim LiDAR point cloud to a 2D depth image using (y,z) projection.
            
            points: np.array of shape (N, 3)
            img_h, img_w: desired output image size
            max_range: value used to normalise distance (paper uses 100m)
            """
            
            # Step 1: Compute distance of each point from drone
            distances = np.linalg.norm(points, axis=1)
            
            # Step 2: Normalise distances to [0,1]
            norm_dist = np.clip(distances / max_range, 0.0, 1.0)
            
            # Extract y and z coordinates
            y = points[:, 1]
            z = points[:, 2]
            
            # Step 3: Convert y,z to image pixel indices ------------------
            
            # Map y coordinate range [-max_range, +max_range] → [0, img_w)
            y_idx = ((y + max_range) / (2 * max_range)) * (img_w - 1)
            
            # Map z coordinate range [-max_range, +max_range] → [0, img_h)
            z_idx = ((z + max_range) / (2 * max_range)) * (img_h - 1)
            
            # Convert to integer pixel indices
            y_idx = np.int32(np.clip(y_idx, 0, img_w - 1))
            z_idx = np.int32(np.clip(z_idx, 0, img_h - 1))
            
            # Step 4: Create depth image and place values
            depth_image = np.ones((img_h, img_w), dtype=np.float32)  # initialize with 1 (max range)
            
            for i in range(len(points)):
                depth_image[z_idx[i], y_idx[i]] = norm_dist[i]
            
            
            return depth_image


        # Project to 2D depth image (simple approach)
        depth_img = lidar_to_depth_image(pts)
        # Resize to HxW
        depth_img = depth_img.reshape(self.H, self.W)

        return depth_img.astype(np.float32)


    # ------------------------------------------------------------------
    # DISTANCE + BEARING TO GOAL
    # ------------------------------------------------------------------
    def get_distance_to_goal(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        drone = np.array([pos.x_val, pos.y_val, pos.z_val])
        goal = np.array([self.SX + self.goal[0], self.SY + self.goal[1], self.SZ + self.goal[2]])
        return np.linalg.norm(drone - goal)

    def get_bearing(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        dx = (self.SX + self.goal[0]) - pos.x_val
        dy = (self.SY + self.goal[1]) - pos.y_val
        return float(math.atan2(dy, dx))


    # ------------------------------------------------------------------
    # TERMINATION CHECK
    # ------------------------------------------------------------------
    def check_termination(self):

        # Check collision
        if self.client.getMultirotorState().collision.has_collided:
            return True, False, False, False

        # Goal reached
        if self.get_distance_to_goal() < 1.0:
            return True, False,False, True
        
        # Out of bounds check
        pos = self.client.getMultirotorState().kinematics_estimated.position
        x, y, z = pos.x_val, pos.y_val, pos.z_val

        # X bounds (maze corridor from x=0 to x=35)
        if x < self.SX - 5 or x > self.SX + 40:
            return True, False, False, False

        # Y bounds (walls at y = ±4 → give margin to ±5)
        if y < self.SY - 5 or y > self.SY + 5:
            return True, False, False, False

        # Z bounds (walls are about ~2m high)
        if z < -1.0 or z > 1.5:   # center-based limits
            return True, False, False, False
        
        # Timeout
        if self.steps >= self.max_steps:
            return False, True, False, False

        return False, False, False, False


    def render(self, mode="human"):
        pass

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
