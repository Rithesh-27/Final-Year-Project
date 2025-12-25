import airsim
import numpy as np

def build_course(client, DRONE_X=500, DRONE_Y=500):

    tele_pose = airsim.Pose(
        airsim.Vector3r(DRONE_X, DRONE_Y, -1.5),
        airsim.to_quaternion(0, 0, 0)
    )

    client.simSetVehiclePose(tele_pose, ignore_collision=True)

    # -----------------------------------------------------
    # Helper to spawn objects relative to new sandbox
    # -----------------------------------------------------
    def spawn(name, asset, dx, dy, dz, sx=1, sy=1, sz=1, yaw=0):
        pose = airsim.Pose(
            airsim.Vector3r(DRONE_X + dx, DRONE_Y + dy, dz),
            airsim.to_quaternion(0, 0, yaw)
        )
        scale = airsim.Vector3r(sx, sy, sz)
        client.simSpawnObject(name, asset, pose, scale)

    # -----------------------------------------------------
    # Obstacle 1: Wider right turn walls
    # -----------------------------------------------------
    spawn("wall1a", "Cube", 5,   0, 0, sx=3, sy=0.8, sz=5)
    spawn("wall1b", "Cube", 5,   5, 0, sx=3, sy=0.8, sz=5)
    spawn("wall1c", "Cube", 5,  -5, 0, sx=3, sy=0.8, sz=5)

    # -----------------------------------------------------
    # Obstacle 2: Wider gap
    # -----------------------------------------------------
    spawn("gap1a", "Cube", 11,  5, 0, sx=2, sy=0.8, sz=5)
    spawn("gap1b", "Cube", 11, -4, 0, sx=2, sy=0.8, sz=5)

    # -----------------------------------------------------
    # Obstacle 3: Wider left turn
    # -----------------------------------------------------
    spawn("turn1a", "Cube", 17,  5, 0, sx=3, sy=0.8, sz=5)
    spawn("turn1b", "Cube", 17,  0, 0, sx=3, sy=0.8, sz=5)
    spawn("turn1c", "Cube", 17, -5, 0, sx=3, sy=0.8, sz=5)

    # -----------------------------------------------------
    # Obstacle 4: Gentler zig-zag
    # -----------------------------------------------------
    spawn("zig1", "Cube", 23,  2, 0, sx=2, sy=0.8, sz=5)
    spawn("zig2", "Cube", 27, -2, 0, sx=2, sy=0.8, sz=5)
    spawn("zig3", "Cube", 31,  2, 0, sx=2, sy=0.8, sz=5)

    # -----------------------------------------------------
    # Final block (kept wide)
    # -----------------------------------------------------
    spawn("finalblock", "Cube", 36, 0, 0, sx=3, sy=0.8, sz=5)

    # -----------------------------------------------------
    # Goal
    # -----------------------------------------------------
    spawn("GOAL", "Cylinder", 42, 0, 0, sx=0.5, sy=0.5, sz=2)

    print("Wider, easier course ready!")



