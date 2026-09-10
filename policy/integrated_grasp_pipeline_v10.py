"""Integrated grasp pipeline v10: capture/detect/grasp until the box is empty."""

import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

# Reuse the unchanged and field-tested robot motion functions from v9.
from integrated_grasp_pipeline_v9 import *  # noqa: F401,F403
from zerith_client_v10 import (
    create_client as create_v10_client,
    detect_parts as detect_parts_v10,
)


MAX_SORTING_ROUNDS = int(os.environ.get("ZERITH_MAX_SORTING_ROUNDS", "10"))
ROUND_SETTLE_SECONDS = float(os.environ.get("ZERITH_ROUND_SETTLE_SECONDS", "2.0"))


def perceive_one_round(rgb_path, depth_path, debug_dir):
    """Detect/register one fresh frame and return (status, pose items)."""
    os.makedirs(debug_dir, exist_ok=True)
    client = create_v10_client(ZMQ_SERVER_ADDR)
    if client is None:
        return "error", []

    try:
        color, boxes, error = detect_parts_v10(client, rgb_path)
        if error is not None:
            print(f" -> [错误] Detection 调用失败: {error}")
            return "error", []
        if not boxes:
            return "empty", []

        save_detection_debug(debug_dir, color, boxes)
        depth = np.load(depth_path)
        category_counts = {}
        items = []

        for detection_index, box_dict in enumerate(boxes):
            label = box_dict["label"]
            category_id = str(box_dict["category_id"])
            instance_index = category_counts.get(category_id, 0)
            category_counts[category_id] = instance_index + 1
            box = [int(box_dict[k]) for k in ("x1", "y1", "x2", "y2")]
            output_dir = os.path.join(debug_dir, f"{category_id}_{instance_index}")
            print(
                f" -> [{detection_index + 1}/{len(boxes)}] "
                f"处理 {category_id}_{instance_index}: {label}"
            )
            success = process_label(
                client, K_COLOR, color, depth, label, category_id, box,
                None, None, output_dir, REGISTER_ITERATIONS, False,
            )
            pose_path = build_pose_path(debug_dir, category_id, instance_index)
            if success and os.path.isfile(pose_path):
                items.append((category_id, instance_index, pose_path))
            else:
                print(f" -> [警告] {category_id}_{instance_index} 注册失败，跳过本次抓取。")
        return "parts", items
    except Exception as exc:
        print(f" -> [异常] 本轮感知失败: {exc}")
        return "error", []
    finally:
        client.close()


def execute_round(robot, items):
    """Execute all valid grasp commands from one snapshot."""
    executed = 0
    normal_items = [item for item in items if item[0] != "cat4"]
    cat4_items = [item for item in items if item[0] == "cat4"]

    for category_id, instance_index, pose_path in normal_items:
        print(f"\n[本轮抓取] {category_id}_{instance_index}")
        selection = select_arm(robot, pose_path)
        if selection is None:
            continue
        flag, target_pos, angle = selection
        target_quat = R.from_euler("xyz", [angle, 0, 0], degrees=True).as_quat()
        if flag == 0:
            grasp_by_left(robot, target_pos, target_quat)
        else:
            grasp_by_right(robot, target_pos, target_quat)
        executed += 1

    for category_id, instance_index, pose_path in cat4_items:
        print(f"\n[本轮抓取 cat4] {category_id}_{instance_index}")
        selection = select_arm(robot, pose_path)
        if selection is None:
            continue
        _, target_pos, angle = selection
        target_quat = R.from_euler("xyz", [angle, 0, 0], degrees=True).as_quat()
        grasp_by_right1(robot, target_pos, target_quat, "cat4")
        executed += 1
    return executed


def main():
    robot = H1Robot()
    completed = False
    total_commands = 0
    completed_rounds = 0
    try:
        print("[INIT] 实例化机器人并连接...")
        if not robot.robot_connect():
            print("连接机器人失败！")
            return
        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()
        chassis_move(robot, 0.8)
        time.sleep(1.0)
        prepare_robot_posture(robot, 0, 0, 0.67, 1.2)
        arm_move_pre(
            robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.02, 0.30], [0.0, 0.0, 0.0, 1.0],
        )
        chassis_move(robot, 0.3)
        time.sleep(2.0)

        for round_index in range(1, MAX_SORTING_ROUNDS + 1):
            print(f"\n========== 分拣循环 {round_index}/{MAX_SORTING_ROUNDS} ==========")
            rgb_path, depth_path = capture_rgbd_data()
            round_debug_dir = os.path.join(CLIENT_DEBUG_DIR, f"round_{round_index:02d}")
            status, items = perceive_one_round(rgb_path, depth_path, round_debug_dir)

            if status == "error":
                print("[终止] 本轮感知失败，不能把失败误判为盒子已空。")
                break
            if status == "empty":
                completed = True
                print(
                    "\n[完成] 服务端返回 0 个零件，"
                    "机器人结束分拣循环。"
                )
                break
            if not items:
                print("[终止] 检测到零件，但所有位姿注册均失败；请检查后重试。")
                break

            total_commands += execute_round(robot, items)
            completed_rounds += 1
            time.sleep(ROUND_SETTLE_SECONDS)
        else:
            print(
                f"[停止] 已达到最大循环次数 {MAX_SORTING_ROUNDS}，"
                "尚未确认盒子为空。"
            )

        if completed:
            print(
                f"[机器人状态] 所有零件分拣成功；"
                f"共执行 {completed_rounds} 轮、{total_commands} 条抓取命令。"
            )
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[!] 捕获到 Ctrl+C 中断，准备安全下线。")
    except Exception as exc:
        print(f"\n[!] 运行时发生异常: {exc}")
    finally:
        print("[清理] 回收机器人控制权...")
        if hasattr(robot, "robot_deinit"):
            robot.robot_deinit()


if __name__ == "__main__":
    main()
