import airsim
import time
import os
import numpy as np
import cv2

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


# ---------------------------------------------
# CONNECT TO AIRSIM
# ---------------------------------------------
client = airsim.MultirotorClient()
client.confirmConnection()

# enable API control
client.enableApiControl(True)
client.armDisarm(True)

# takeoff
client.takeoffAsync().join()

print(client.simListSceneObjects())

print("Drone is ready!")

# -----------------------------------------------
# CREATE OUTPUT FOLDER
# -----------------------------------------------



# -------------------------------------
# LAND DRONE
# -------------------------------------
client.landAsync(vehicle_name="Drone1").join()
client.armDisarm(False, "Drone1")
client.enableApiControl(False, "Drone1")
