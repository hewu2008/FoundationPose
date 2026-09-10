"""
@file get_rgbd_grpc_headless.py
@brief 基于 gRPC 客户端的 RGB-D (彩色+深度) 图像获取与保存
* [功能介绍]
1. 通过 CameraClient 同时从服务端获取彩色图 (BGR) 和深度图数据。
2. 自动抓取第一帧有效对齐数据并保存。
3. 【关键修改】将保存的深度矩阵强制转换为 float32 格式，且单位转换为米 (m)，直接适配 FoundationPose。
4. 无需图形界面，运行完毕后自动安全退出。
"""

import os
import sys
import cv2
import numpy as np
import time

# 路径兼容：将 src 目录加入搜索路径以导入 camera_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from camera_client import CameraClient

def main():
    # ================= 配置参数 =================
    GRPC_TARGET = "localhost:50051"  
    CAMERA_NAME = "rs/cam_high"  

    print(f"[*] 正在连接 gRPC 服务端: {GRPC_TARGET} ...")
    client = CameraClient(grpc_target=GRPC_TARGET, enable_depth=True)
    
    try:
        client.start()
        print(f"[*] 已连接。正在同步等待 {CAMERA_NAME} 的彩色流和深度流...")

        max_retries = 50
        
        for i in range(max_retries):
            # 1. 使用官方 API 获取数据
            depth_data = client.get_latest_depth(CAMERA_NAME)
            color_data = client.get_latest_frame(CAMERA_NAME) 
            
            # 2. 只有当两个数据流都成功拿到时，才进行处理
            if depth_data is not None and color_data is not None:
                depth_raw_mm, d_timestamp = depth_data
                color_raw, c_timestamp = color_data # 获取到的已经是 BGR 格式
                
                # ================= 核心修改区 =================
                # 将 uint16 的毫米数据转换为 float32 的米，专门喂给算法模型
                depth_raw_m = depth_raw_mm.astype(np.float32) / 1000.0
                
                # --- 深度图可视化处理 (仅为保存 png 供肉眼查看) ---
                # 用原始毫米数据来做伪彩色映射效果更好
                depth_viz = (depth_raw_mm * 0.03).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

                # ================= 保存文件 =================
                time_str = time.strftime("%Y%m%d_%H%M%S")
                
                img_filename = f"zerith_rgb.png"
                depth_img_filename = f"depth_clean_{time_str}.png"
                depth_raw_filename = f"zerith_depth.npy" # 改了名字以示区分
                
                # 1. 保存彩色图
                cv2.imwrite(img_filename, color_raw)
                
                # 2. 保存伪彩色深度图
                cv2.imwrite(depth_img_filename, depth_color)
                
                # 3. 保存给 FoundationPose 用的浮点数(米)深度矩阵
                np.save(depth_raw_filename, depth_raw_m)
                
                # 计算中心点距离用于终端打印确认
                height, width = depth_raw_m.shape
                center_x, center_y = width // 2, height // 2
                center_depth_val = depth_raw_m[center_y, center_x]
                
                print("\n[+] 🎉 成功捕获并保存了对齐的 RGB-D 图像！")
                print(f" ├── 彩色图像 (BGR)     : {img_filename}")
                print(f" ├── 伪彩色深度图       : {depth_img_filename}")
                print(f" ├── 算法专用深度图 (m) : {depth_raw_filename} (已转为 float32, 单位: 米)")
                print(f" └── 中心点深度值       : {center_depth_val:.3f} 米")
                
                break # 拿到数据并保存后，跳出循环
            
            time.sleep(0.1) # 没拿到就等 100ms 再试
            
        else:
            print(f"\n[-] ❌ 获取图像超时。请检查服务端是否正常输出了 {CAMERA_NAME} 的流。")

    except KeyboardInterrupt:
        print("\n[!] 捕获到 Ctrl+C 中断。")
    except Exception as e:
        print(f"\n[-] ❌ 运行出错: {e}")
        
    finally:
        # 确保安全关闭客户端
        client.stop()
        print("[*] 客户端已停止，网络资源已释放。")

if __name__ == "__main__":
    main()
