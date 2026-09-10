import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def calculate_target_relative_pose(cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam):
    """坐标系变换：感知位姿 -> 手臂相对零点目标位姿"""
    # T1 = np.eye(4)
    # T1[:3, :3] = R.from_quat(cam_quat_rel).as_matrix()
    # T1[:3, 3] = cam_pos_rel

    T2 = np.eye(4)
    T2[:3, :3] = R.from_euler('xyz', [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
    T2[:3, 3] = [0.2194, 0.0325, 0.6075]

    T3 = np.eye(4)
    T3[:3, 3] = [-0.5743, -0.1800, -0.1208]

    # T4_inv = np.eye(4)
    # T4_inv[:3, :3] = R.from_quat(arm_quat_rel).as_matrix()
    # T4_inv[:3, 3] = arm_pos_rel
    # T_comp = np.eye(4)
    # T_comp[:3, 3] = [0.0, 0.0, 0.0]
    # T4_inv = np.dot(T4_inv, T_comp)
    # T4 = np.linalg.inv(T4_inv)

    # T_obj_in_arm = np.dot(T4, np.dot(T3, np.dot(T2, np.dot(T1, T_obj_cam))))

    T_obj_in_arm = np.dot(T3, np.dot(T2, T_obj_cam))

    # T_grasp_local = np.eye(4)
    # T_grasp_local[:3, :3] = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True).as_matrix()
    # T_grasp_local[:3, 3] = [0.0, 0.0, 0.0] 

    # T_final = np.dot(T_obj_in_arm, T_grasp_local)

    T_final = T_obj_in_arm

    target_pos = T_final[:3, 3]
    target_quat = R.from_matrix(T_final[:3, :3]).as_quat()

    return target_pos, target_quat

def load_pose_matrix(filepath):
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            if row: matrix.append(row)
    return np.array(matrix)

def main():
    T_obj_cam = load_pose_matrix("1.txt")
    target_pos, _ = calculate_target_relative_pose([0, 0, 0], [0, 0, 0, 1],
                                                [0, 0, 0], [0, 0, 0, 1],
                                                T_obj_cam)
    print(target_pos)

if __name__ == "__main__":
    main()