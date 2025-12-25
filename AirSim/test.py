import airsim
import matplotlib.pyplot as plt
import numpy as np
import time
from set_course import build_course
from blocks_script import start_blcoks,stop_blocks

def lidar_to_depth_image(
    pts,
    num_channels=64,
    h_bins=256,
    v_fov_up=15.0,
    v_fov_down=-15.0,
    max_range=50.0
):
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

def visualize_depth(depth):
    """
    depth: (H, W) normalized to [0,1]
    """
    plt.figure(figsize=(10, 3))
    # plt.imshow(depth, cmap="gray", aspect="auto")
    plt.imshow(1.0 - depth, cmap="inferno", aspect="auto")
    plt.colorbar(label="Normalized Depth")
    plt.xlabel("Azimuth (left → right)")
    plt.ylabel("Vertical Channels (top → bottom)")
    plt.title("LiDAR Depth Image")
    plt.show()


# ---------------------------------------------
# CONNECT TO AIRSIM
# ---------------------------------------------
start_blcoks()
client = airsim.MultirotorClient()
client.confirmConnection()

# enable API control
client.enableApiControl(True)
client.armDisarm(True)

build_course(client)

print("Drone is ready!")
time.sleep(0.1)

_ = client.getLidarData(lidar_name="Lidar1", vehicle_name="Drone1") # Flush old points
time.sleep(0.05)

# -----------------------------------------------
# CREATE OUTPUT FOLDER
# -----------------------------------------------
lidar = client.getLidarData(lidar_name="Lidar1", vehicle_name="Drone1")
pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1,3)

depth_image = lidar_to_depth_image(pts)
image = visualize_depth(depth_image)

# -------------------------------------
# LAND DRONE
# -------------------------------------
client.landAsync(vehicle_name="Drone1").join()
client.armDisarm(False, "Drone1")
client.enableApiControl(False, "Drone1")

stop_blocks()
