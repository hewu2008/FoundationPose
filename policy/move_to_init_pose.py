import os
import sys
import time
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R, Slerp

# ================= SDK import & environment setup =================
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

# Robot SDK
from lib.lib_h1_sdk_python import (
    H1Robot, MotorControlMode,
    ArmAction, ArmPose, ArmEndPose
)

# ================= System control parameters =================
RATE_HZ = 500             # Low-level control frequency 500Hz
DT = 1.0 / RATE_HZ        # Control period 0.002s

# Initial observation pose targets (same as main() in integrated_grasp_pipeline_v18.py)
WAIST_INIT_Z = 0.67       # Waist height
WAIST_INIT_PITCH = 1.2    # Waist pitch angle
ARM_INIT_XYZ = [0.0, 0.03, 0.28]     # Arm target position (relative to zero; left arm y positive, right arm y negative)
ARM_INIT_QUAT = [0.0, 0.0, 0.0, 1.0]  # Arm target orientation (quaternion xyzw)

# ================= 1. Robot posture preparation =================
def prepare_robot_posture(robot, cur_waist_z, cur_waist_pitch, tar_waist_z, tar_waist_pitch):
    print("\n[Step A] Adjusting robot initial observation posture...")
    waist_steps = int(3.0 * RATE_HZ)

    for i in range(1, waist_steps + 1):
        ratio = i / waist_steps
        waist_pose = ArmPose()
        waist_pose.x = 0.0
        waist_pose.y = 0.0
        waist_pose.roll = 0.0
        waist_pose.yaw = 0.0

        waist_pose.z    = cur_waist_z    + (tar_waist_z    - cur_waist_z)    * ratio
        waist_pose.pitch = cur_waist_pitch + (tar_waist_pitch - cur_waist_pitch) * ratio

        end_pose = robot.armPoseToArmEndPose(waist_pose)
        robot.setWaist_high(end_pose)
        time.sleep(DT)

    time.sleep(1.5)
    print(" -> Observation posture adjusted.")

def _move_arm(robot, arm, start_xyz, start_quat, dest_xyz, dest_quat,
              duration=4.0, rate=RATE_HZ, dt=DT):
    """Smooth interpolation for a single arm, called by threads"""
    steps = int(duration * rate)
    sx, sy, sz = start_xyz
    dx, dy, dz = dest_xyz

    key_rots = R.from_quat([start_quat, dest_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(1, steps + 1):
        ratio = i / steps
        x = sx + (dx - sx) * ratio
        y = sy + (dy - sy) * ratio
        z = sz + (dz - sz) * ratio
        quat = slerp(ratio).as_quat()

        pose = ArmEndPose()
        pose.position = [x, y, z]
        pose.rotation = [quat[0], quat[1], quat[2], quat[3]]

        robot.setArm_high(arm, pose)
        time.sleep(dt)

def arm_move_pre(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
    print("\n[Step E] Moving both arms smoothly toward the target simultaneously...")

    # Different targets for left/right arms (y values are negatives of each other)
    left_dest = [dest_xyz[0],  dest_xyz[1], dest_xyz[2]]
    right_dest = [dest_xyz[0], -dest_xyz[1], dest_xyz[2]]   # negate y

    # Create threads: left and right arms move simultaneously
    t_left = threading.Thread(
        target=_move_arm,
        args=(robot, ArmAction.LEFT_ARM, cur_xyz, cur_quat,
              left_dest, dest_quat)
    )
    t_right = threading.Thread(
        target=_move_arm,
        args=(robot, ArmAction.RIGHT_ARM, cur_xyz, cur_quat,
              right_dest, dest_quat)
    )

    t_left.start()
    t_right.start()

    # Wait for both threads to finish
    t_left.join()
    t_right.join()

    time.sleep(0.5)
    print(" -> Both arms have reached their targets smoothly!")

# ================= Move to initial pose =================
def move_to_init_pose(robot):
    """Move the robot to the initial observation pose used in integrated_grasp_pipeline_v18.py"""
    print("\n[INIT] Moving to initial pose...")

    # 1. Move waist to the initial height/pitch
    prepare_robot_posture(robot, 0.0, 0.0, WAIST_INIT_Z, WAIST_INIT_PITCH)
    # 2. Move both arms to the initial position
    arm_move_pre(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                 ARM_INIT_XYZ, ARM_INIT_QUAT)
    time.sleep(1.5)

    print("\n[DONE] Reached initial pose: "
          f"waist z={WAIST_INIT_Z}, pitch={WAIST_INIT_PITCH}; "
          f"arms xyz={ARM_INIT_XYZ}")

# ================= Main control flow =================
def main():
    robot = H1Robot()
    try:
        print("[INIT] Instantiating robot and connecting...")
        if not robot.robot_connect():
            print("Failed to connect to robot!")
            return

        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()
        time.sleep(3.0)

        # Move to initial pose
        time.sleep(1.5)
        move_to_init_pose(robot)

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected, performing safe shutdown...")
    except Exception as e:
        print(f"\n[!] Runtime exception: {e}")

    finally:
        print("[Cleanup] Releasing robot control...")
        if 'robot' in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()

if __name__ == "__main__":
    main()
