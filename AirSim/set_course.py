import airsim

def build_course(client, DRONE_X=500, DRONE_Y=500):

    tele_pose = airsim.Pose(
        airsim.Vector3r(DRONE_X, DRONE_Y, 0),
        airsim.to_quaternion(0, 0, 0)
    )

    client.simSetVehiclePose(tele_pose, ignore_collision=True)

    print("Drone moved to new sandbox area.")

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
    # Spawn obstacle course relative to drone
    # -----------------------------------------------------

    print("Spawning wider course...")

    # --------- Obstacle 1: Wider right turn walls ----------
    spawn("wall1a", "Cube", 5, 0, 0,  sx=3, sy=1, sz=2)
    spawn("wall1b", "Cube", 5, 4, 0,  sx=3, sy=1, sz=2)
    spawn("wall1c", "Cube", 5, -4, 0, sx=3, sy=1, sz=2)

    # --------- Obstacle 2: Wider narrow gap ----------
    spawn("gap1a", "Cube", 10, 3, 0, sx=2, sy=1, sz=2)
    spawn("gap1b", "Cube", 10, -3, 0, sx=2, sy=1, sz=2)

    # --------- Obstacle 3: Wider left turn ----------
    spawn("turn1a", "Cube", 15, 5, 0, sx=3, sy=1, sz=2)
    spawn("turn1b", "Cube", 15, 0, 0, sx=3, sy=1, sz=2)
    spawn("turn1c", "Cube", 15, -5, 0, sx=3, sy=1, sz=2)

    # --------- Obstacle 4: Wider zig-zag ----------
    spawn("zig1", "Cube", 20,  4, 0, sx=2, sy=1, sz=2)
    spawn("zig2", "Cube", 23, -3, 0, sx=2, sy=1, sz=2)
    spawn("zig3", "Cube", 26,  4, 0, sx=2, sy=1, sz=2)

    # --------- Final block ----------
    spawn("finalblock", "Cube", 30, 0, 0, sx=3, sy=1, sz=2)

    # --------- Goal cylinder ----------
    spawn("GOAL", "Cylinder", 35, 0, 0, sx=2, sy=2, sz=2)

    print("Wider course ready in sandbox area!")
