import os
import sys
import time
import cv2
import zmq
import pickle
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# ================= 导入 SDK 及环境配置 =================
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

# 机器人 SDK
from lib.lib_h1_sdk_python import (
    H1Robot, MotorControlMode, EtherCAT_Motor_Index,
    ArmAction, ArmPose, ArmEndPose, Motor_Control
)

# 相机 gRPC 客户端 (需确保与本脚本路径层级兼容)
from camera_client import CameraClient

# ================= 系统控制参数 =================
RATE_HZ = 500             # 底层控制频率 500Hz
DT = 1.0 / RATE_HZ        # 控制周期 0.002s

# FoundationPose & Zerith 配置
GRPC_TARGET = "localhost:50051"
CAMERA_NAME = "rs/cam_high"
ZMQ_SERVER_ADDR = "tcp://172.31.200.250:5555"
MESH_FILE = "assets/DPPUB-204001196-AAX_01_01.obj"
TARGET_LABELS = ["white translucent plastic brake fluid reservoir with a blue or black cap"]

# ================= 1. ZMQ 客户端类 =================
class ZerithFoundationPoseClient:
    def __init__(self, server_addr):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(server_addr)
        print(f"[ZMQ] 已连接到算法服务端: {server_addr}")
    
    def send_request(self, command, **kwargs):
        request = {'command': command}
        request.update(kwargs)
        try:
            self.socket.send(pickle.dumps(request))
            response = self.socket.recv()
            return pickle.loads(response)
        except Exception as e:
            print(f"[ZMQ错误] 请求发送失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def register(self, K, rgb, depth, labels=None, threshold=0.3, iteration=5):
        params = {
            'K': K, 'rgb': rgb, 'depth': depth,
            'labels': labels if labels else ["object."],
            'threshold': threshold, 'iteration': iteration
        }
        return self.send_request('register', **params)
    
    def track(self, K, rgb, depth, iteration=2):
        params = {
            'K': K, 'rgb': rgb, 'depth': depth, 'iteration': iteration
        }
        return self.send_request('track', **params)
    
    def ping(self):
        return self.send_request('ping')
    
    def close(self):
        self.socket.close()
        self.context.term()

# ================= 2. 机器人姿态准备模块 =================
def prepare_robot_posture(robot, cur_waist_z, cur_waist_pitch, tar_waist_z, tar_waist_pitch):
    print("\n[流程 A] 正在调整机器人初始观测姿态...")
    waist_steps = int(3.0 * RATE_HZ)

    for i in range(1, waist_steps + 1):
        ratio = i / waist_steps
        waist_pose = ArmPose()
        waist_pose.x = 0.0
        waist_pose.y = 0.0
        waist_pose.roll = 0.0
        waist_pose.yaw = 0.0

        # 关键修改：从当前值平滑插值到目标值
        waist_pose.z    = cur_waist_z    + (tar_waist_z    - cur_waist_z)    * ratio
        waist_pose.pitch = cur_waist_pitch + (tar_waist_pitch - cur_waist_pitch) * ratio

        end_pose = robot.armPoseToArmEndPose(waist_pose)
        robot.setWaist_high(end_pose)
        time.sleep(DT)

    time.sleep(1.5)

    # # 动作 2：头部俯视
    # print(" -> 2/2 头部云台俯视中...")
    # head_pitch = Motor_Control()
    # head_pitch.Position = 0.0
    # target_pitch_angle = 0.3  
    # head_steps = int(1.5 * RATE_HZ) 
    
    # for i in range(1, head_steps + 1):
    #     head_pitch.Position = target_pitch_angle * (i / head_steps)
    #     robot.setHead_high(EtherCAT_Motor_Index.MOTOR_HEAD_UP, head_pitch)
    #     time.sleep(DT)
        
    # time.sleep(2.0)
    print(" -> 观测姿态调整完毕。")

def arm_move(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
    print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
    duration_arm = 4.0
    steps_arm = int(duration_arm * RATE_HZ)

    start_x, start_y, start_z = cur_xyz[0], cur_xyz[1], cur_xyz[2]
    start_quat = cur_quat

    dest_x, dest_y, dest_z = dest_xyz[0], dest_xyz[1], dest_xyz[2] 
    # dest_quat = [0.0, 0.0, 0.0, 1.0]
    # dest_quat = [0.0000, -0.2588, 0.0000, 0.9659]

    key_rots = R.from_quat([start_quat, dest_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(1, steps_arm + 1):
        ratio = i / steps_arm

        x = start_x + (dest_x - start_x) * ratio
        y = start_y + (dest_y - start_y) * ratio
        z = start_z + (dest_z - start_z) * ratio

        interp_quat = slerp(ratio).as_quat()

        target_end_pose = ArmEndPose()
        target_end_pose.position = [x, y, z]
        target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

        robot.setArm_high(ArmAction.LEFT_ARM, target_end_pose)
        robot.setArm_high(ArmAction.RIGHT_ARM, target_end_pose)
        time.sleep(DT)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

# ================= 3. RGB-D 获取模块 =================
def capture_rgbd_data():
    """获取 gRPC 图像并保存"""
    print(f"\n[流程 B] 正在连接相机服务获取 RGB-D 数据 ({GRPC_TARGET})...")
    client = CameraClient(grpc_target=GRPC_TARGET)
    client.start()
    
    rgb_path = "zerith_rgb.png"
    depth_path = "zerith_depth.npy"
    
    try:
        max_retries = 50
        for i in range(max_retries):
            depth_data = client.get_latest_depth(CAMERA_NAME)
            color_data = client.get_latest_frame(CAMERA_NAME) 
            
            if depth_data is not None and color_data is not None:
                depth_raw_mm, _ = depth_data
                color_raw, _ = color_data
                
                # 转换为算法需要的 float32 (米)
                depth_raw_m = depth_raw_mm.astype(np.float32) / 1000.0
                
                cv2.imwrite(rgb_path, color_raw)
                np.save(depth_path, depth_raw_m)
                print(f" -> 捕获成功！已保存 {rgb_path} 和 {depth_path}")
                return rgb_path, depth_path
                
            time.sleep(0.1)
        raise TimeoutError("获取图像超时！请检查 gRPC 节点。")
    finally:
        client.stop()

# ================= 4. 感知端逻辑 (Register + Track) =================
def run_perception_client(rgb_path, depth_path, mesh_file, labels):
    """按照原定逻辑：1次 Register + 9次 Track，最后保存位姿"""
    print("\n[流程 C] 请求 Zerith FoundationPose 推理位姿 (1 Register + 9 Track)...")
    
    try:
        mesh = trimesh.load(mesh_file)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        print(f" -> 已加载 Mesh: {mesh_file}")
    except Exception as e:
        print(f" -> [警告] Mesh 加载失败 (不影响推理): {e}")

    client = ZerithFoundationPoseClient(ZMQ_SERVER_ADDR)
    
    response = client.ping()
    if response['status'] != 'success':
        print(f" -> [错误] 服务端连接失败: {response.get('message', '未知错误')}")
        return False
    print(" -> 服务端连接成功！开始推理...")
    
    try:
        final_pose = None
        for i in range(10):
            color = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth = np.load(depth_path)

            K_color = np.array([
                [607.62, 0.00,  329.68],
                [0.00,  608.40, 243.36],
                [0.00,  0.00, 1.00]
            ])
            
            if i == 0:
                print(f" -> [第 {i} 帧] 正在进行注册 (Registering)...")
                response = client.register(
                    K=K_color, rgb=color, depth=depth,
                    labels=labels, threshold=0.41, iteration=5
                )
                if response['status'] != 'success':
                    print(f" -> [错误] 注册失败: {response.get('message', '未知错误')}")
                    return False
                pose = response['pose']
                if hasattr(pose, 'numpy'): pose = pose.numpy()
                print(" -> 注册成功！")
            else:
                start_time = time.time()
                response = client.track(
                    K=K_color, rgb=color, depth=depth, iteration=2
                )
                if response['status'] != 'success':
                    print(f" -> [错误] 跟踪失败: {response.get('message', '未知错误')}")
                    return False
                pose = response['pose']
                if hasattr(pose, 'numpy'): pose = pose.numpy()
                end_time = time.time()
                print(f" -> [第 {i} 帧] 跟踪成功，耗时: {end_time - start_time:.4f}s")
            
            final_pose = pose

        if final_pose is not None:
            np.savetxt("0.txt", final_pose.reshape(4, 4))
            print("\n -> 🎉 感知推理(Register+Track)全部完成！已将最终 4x4 矩阵写入 0.txt。")
            return True
            
        return False

    except Exception as e:
        print(f" -> [异常] 感知端运行崩溃: {e}")
        return False
        
    finally:
        client.close()

# ================= 5. 抓取运算与执行模块 =================
def load_pose_matrix(filepath):
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            if row: matrix.append(row)
    return np.array(matrix)

def calculate_target_relative_pose(cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam):
    """坐标系变换：感知位姿 -> 手臂相对零点目标位姿"""
    T1 = np.eye(4)
    T1[:3, :3] = R.from_quat(cam_quat_rel).as_matrix()
    T1[:3, 3] = cam_pos_rel

    T2 = np.eye(4)
    T2[:3, :3] = R.from_euler('xyz', [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
    T2[:3, 3] = [0.2194, 0.0325, 0.6075]

    T3 = np.eye(4)
    T3[:3, 3] = [-0.5743, -0.1800, -0.1208]

    T4_inv = np.eye(4)
    T4_inv[:3, :3] = R.from_quat(arm_quat_rel).as_matrix()
    T4_inv[:3, 3] = arm_pos_rel
    T_comp = np.eye(4)
    
    #T_comp[:3, 3] = [0.03, -0.02, 0.0] #0.08
    #T_comp[:3, 3] = [0.12, -0.01, 0.12]
    T4_inv = np.dot(T4_inv, T_comp)
    T4 = np.linalg.inv(T4_inv)

    T_obj_in_arm = np.dot(T4, np.dot(T3, np.dot(T2, np.dot(T1, T_obj_cam))))

    T_grasp_local = np.eye(4)
    T_grasp_local[:3, :3] = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True).as_matrix()
    T_grasp_local[:3, 3] = [0.0, 0.0, 0.0] 

    T_final = np.dot(T_obj_in_arm, T_grasp_local)

    target_pos = T_final[:3, 3]
    target_quat = R.from_matrix(T_final[:3, :3]).as_quat()

    return target_pos, target_quat

def chassis_move(robot: H1Robot):
    """
    以0.2m/s的速度前进1米
    """
    DT = 0.2  # 200ms发送一次指令
    SPEED = 0.2  # 目标速度 m/s
    DISTANCE = 1.0  # 目标距离 米
    DURATION = DISTANCE / SPEED  # 持续时间 = 5秒
    
    print(f"[Chassis] 开始前进: 速度={SPEED}m/s, 距离={DISTANCE}m, 持续{DURATION}s")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            
            # 发送底盘速度指令
            robot.setChassis_high(SPEED, 0.0)  # speed_x=0.2, speed_yaw=0.0
            
            # 计算已经过时间
            elapsed_time = time.time() - start_time
            remaining_distance = DISTANCE - (SPEED * elapsed_time)
            print(f"[Chassis] 已过{elapsed_time:.2f}s, 剩余距离{remaining_distance:.2f}m")
            
            # 维持0.2s间隔
            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # 到达目标位置，停止
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 已到达目标位置，前进1米完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 手动停止")
        robot.setChassis_high(0.0, 0.0)  # 停止

def rotate_90_degrees(robot: H1Robot, direction="left"):
    """
    旋转90度
    底盘角速度单位是 rad/s
    
    Args:
        direction: "left" 或 "right"
    """
    DT = 0.2
    ANGULAR_SPEED = 0.5  # 角速度 rad/s (可根据需要调整)
    TARGET_ANGLE = 1.5708  # 90度 = π/2 弧度
    DURATION = TARGET_ANGLE / ANGULAR_SPEED + 1.2  # ≈ 3.14秒
    
    # 确定旋转方向
    if direction == "left":
        yaw_speed = ANGULAR_SPEED  # 正值为左转
    else:
        yaw_speed = -ANGULAR_SPEED  # 负值为右转
    
    print(f"[Chassis] 旋转90度{direction}, 角速度{ANGULAR_SPEED}rad/s, 持续{DURATION:.2f}s")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            
            # 设置角速度，线速度为0
            robot.setChassis_high(0.0, yaw_speed)
            
            elapsed = time.time() - start_time
            rotated_angle = ANGULAR_SPEED * elapsed
            rotated_degrees = rotated_angle * 180 / 3.14159
            print(f"[Chassis] 已旋转: {rotated_degrees:.1f}°")
            
            # 维持0.2s间隔
            elapsed_loop = time.perf_counter() - loop_start
            sleep_time = DT - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 旋转90度完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 中断")
        robot.setChassis_high(0.0, 0.0)

# ================= 主控制流 =================
def main():
    robot = H1Robot()
    try:
        print("[INIT] 实例化机器人并连接...")
        if not robot.robot_connect():
            print("连接机器人失败！")
            return

        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()
        time.sleep(3.0)

        # ------------------------------------------------
        # 1. 机器人姿态准备 (升高、弯腰、低头)
        # ------------------------------------------------
        #chassis_move(robot)
        time.sleep(1.5)
        prepare_robot_posture(robot, 0, 0, 0.6, 0)
        prepare_robot_posture(robot, 0.6, 0, 0.6, 2.4)
        arm_move(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.28],
                [0.0, 0.0, 0.0, 1.0])
        
        prepare_robot_posture(robot, 0.6, 2.4, 0.25, 2.4)
        time.sleep(1.5)
        # ------------------------------------------------
        # 2. 拍摄 RGB-D 照片
        # ------------------------------------------------
        rgb_path, depth_path = capture_rgbd_data()

        # ------------------------------------------------
        # 3. 运行原版感知逻辑并保存 0.txt
        # ------------------------------------------------
        success = run_perception_client(rgb_path, depth_path, MESH_FILE, TARGET_LABELS)
        if not success:
            print("[致命错误] 感知位姿获取失败，终止抓取流程！")
            return

        # ------------------------------------------------
        # 4. 读取手臂/相机的当前状态并解算最终目标点
        # ------------------------------------------------
        print("\n[流程 D] 读取位姿状态并进行矩阵解算...")
        ok_cam, cam_state = robot.getHeadCameraRelative()
        cam_pos_rel = getattr(cam_state, "position", None) or getattr(cam_state, "PositionData", None)
        cam_quat_rel = getattr(cam_state, "rotation", None) or getattr(cam_state, "RotationData", None)

        ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
        arm_pos_rel = getattr(arm_state, "position", None) or getattr(arm_state, "PositionData", None)
        arm_quat_rel = getattr(arm_state, "rotation", None) or getattr(arm_state, "RotationData", None)

        if not (ok_cam and ok_arm):
            print("传感器位姿获取失败！")
            return

        T_obj_cam = load_pose_matrix("0.txt")
        target_pos, target_quat = calculate_target_relative_pose(
            cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam
        )
        print(f" -> 🎯 解算目标相对平移: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

        # ------------------------------------------------
        # 5. 执行手臂平移逼近目标
        # ------------------------------------------------
        print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
        duration_arm = 4.0
        steps_arm = int(duration_arm * RATE_HZ)

        ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
        arm_pos_rel = getattr(arm_state, "position", None)
        start_x, start_y, start_z = arm_pos_rel[0], arm_pos_rel[1], arm_pos_rel[2]
        start_quat = [0.0, 0.0, 0.0, 1.0]
        # start_quat = [0.0000, -0.2588, 0.0000, 0.9659]

        dest_x, dest_y, dest_z = arm_pos_rel[0]+target_pos[0]-0.11, arm_pos_rel[1]+target_pos[1]+0.02, arm_pos_rel[2]+target_pos[2]-0.03
        # dest_quat = [0.0, 0.0, 0.0, 1.0]
        dest_quat = [0.0000, 0.3827, 0.0000, 0.9239]

        key_rots = R.from_quat([start_quat, dest_quat])
        slerp = Slerp([0, 1], key_rots)

        for i in range(1, steps_arm + 1):
            ratio = i / steps_arm

            x = start_x + (dest_x - start_x) * ratio
            y = start_y + (dest_y - start_y) * ratio
            z = start_z + (dest_z - start_z) * ratio

            interp_quat = slerp(ratio).as_quat()

            target_end_pose = ArmEndPose()
            target_end_pose.position = [x, y, z]
            target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

            robot.setArm_high(ArmAction.LEFT_ARM, target_end_pose)
            time.sleep(DT)

        time.sleep(0.5)
        print(" -> 已平滑到达目标点！")

        # ------------------------------------------------
        # 6. 闭合夹爪进行抓取
        # ------------------------------------------------
        print("\n[流程 F] 正在闭合左手夹爪进行抓取...")
        close_cmd = Motor_Control()
        close_cmd.Position = 1.5
        robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, close_cmd)

        time.sleep(2.0)

        # ------------------------------------------------
        # 7. 执行手臂平移至初始位置
        # ------------------------------------------------
        # print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
        # prepare_robot_posture(robot, 0.30, 2.6, 0.30, 0.0)

        # time.sleep(3.0)
        # duration_arm = 4.0
        # steps_arm = int(duration_arm * RATE_HZ)

        # start_x, start_y, start_z = target_pos[0], target_pos[1], target_pos[2]
        # start_quat = [0.0, 0.0, 0.0, 1.0]

        # dest_x, dest_y, dest_z = 0, 0, 0
        # dest_quat = [0.0, 0.0, 0.0, 1.0]

        # key_rots = R.from_quat([start_quat, dest_quat])
        # slerp = Slerp([0, 1], key_rots)

        # for i in range(1, steps_arm + 1):
        #     ratio = i / steps_arm

        #     x = start_x + (dest_x - start_x) * ratio
        #     y = start_y + (dest_y - start_y) * ratio
        #     z = start_z + (dest_z - start_z) * ratio

        #     interp_quat = slerp(ratio).as_quat()

        #     target_end_pose = ArmEndPose()
        #     target_end_pose.position = [x, y, z]
        #     target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

        #     robot.setArm_high(ArmAction.LEFT_ARM, target_end_pose)
        #     time.sleep(DT)

        # time.sleep(2.0)
        # print(" -> 已平滑到达目标点！")

        #rotate_90_degrees(robot, "left")
        time.sleep(2.0)

        #chassis_move(robot)
        time.sleep(2.0)

        # open_cmd = Motor_Control()
        # open_cmd.Position = 0.0
        # robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, open_cmd)

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] 捕获到 Ctrl+C 中断，准备执行安全下线流程...")
    except Exception as e:
        print(f"\n[!] 运行时发生异常: {e}")

    finally:
        print("[清理] 回收机器人控制权...")
        if 'robot' in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()

if __name__ == "__main__":
    main()