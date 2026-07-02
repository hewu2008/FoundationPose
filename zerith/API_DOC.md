# Zerith FoundationPose Server API Documentation

## Overview

Zerith FoundationPose Server 是一个基于 ZeroMQ 的远程姿势估计服务，提供物体检测、注册和跟踪功能。

## 通信协议

- **传输层**: ZeroMQ REQ/REP 模式
- **序列化**: Pickle
- **默认端口**: 5555

## 接口列表

### 1. Ping

**功能**: 检查服务器是否在线

**请求**:
```python
{
    'command': 'ping'
}
```

**响应**:
```python
{
    'status': 'success',
    'message': 'Server is running'
}
```

**客户端调用**:
```python
response = client.ping()
```

---

### 2. Detection

**功能**: 物体部件检测

**请求**:
```python
{
    'command': 'detection',
    'rgb': numpy.ndarray  # RGB格式图像, shape: (H, W, 3), dtype: uint8
}
```

**响应**:
```python
# 成功
{
    'status': 'success',
    'boxes': [
        {
            'label': 'white translucent plastic brake fluid reservoir...',
            'x1': float,   # 左上角X坐标
            'y1': float,   # 左上角Y坐标
            'x2': float,   # 右下角X坐标
            'y2': float    # 右下角Y坐标
        }
    ],
    'message': 'Detection successful, found N boxes'
}

# 失败
{
    'status': 'error',
    'message': 'No boxes detected'
}
```

**客户端调用**:
```python
rgb = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
response = client.detection(rgb)
```

---

### 3. Register

**功能**: 物体注册，估计初始姿态

**请求**:
```python
{
    'command': 'register',
    'K': numpy.ndarray,       # 相机内参矩阵, shape: (3, 3)
    'rgb': numpy.ndarray,     # RGB格式图像, shape: (H, W, 3), dtype: uint8
    'depth': numpy.ndarray,   # 深度图像, shape: (H, W)
    'label': str,             # 检测到的物体标签(可选)
    'box': list,              # 检测框 [x1, y1, x2, y2] (可选)
    'threshold': float,       # 分割阈值, 默认 0.3
    'iteration': int          # 优化迭代次数, 默认 5
}
```

**响应**:
```python
# 成功
{
    'status': 'success',
    'pose': torch.Tensor,     # 4x4 变换矩阵
    'message': 'Registration successful'
}

# 失败
{
    'status': 'error',
    'message': 'Segmentation failed, no valid mask found'
}
```

**客户端调用**:
```python
response = client.register(
    K=K_color,
    rgb=color,
    depth=depth,
    label=label,
    box=[x1, y1, x2, y2],
    threshold=0.34,
    iteration=5
)
```

---

### 4. Track

**功能**: 物体跟踪，更新姿态

**请求**:
```python
{
    'command': 'track',
    'K': numpy.ndarray,       # 相机内参矩阵, shape: (3, 3)
    'rgb': numpy.ndarray,     # RGB格式图像, shape: (H, W, 3), dtype: uint8
    'depth': numpy.ndarray,   # 深度图像, shape: (H, W)
    'iteration': int          # 优化迭代次数, 默认 2
}
```

**响应**:
```python
# 成功
{
    'status': 'success',
    'pose': torch.Tensor,     # 4x4 变换矩阵
    'message': 'Tracking successful'
}

# 失败
{
    'status': 'error',
    'message': 'Tracking failed: ...'
}
```

**客户端调用**:
```python
response = client.track(
    K=K_color,
    rgb=color,
    depth=depth,
    iteration=2
)
```

---

## 完整调用流程

```python
# 1. 创建客户端
client = ZerithFoundationPoseClient('tcp://localhost:5555')

# 2. 检查连接
response = client.ping()
if response['status'] != 'success':
    exit()

# 3. 检测物体
response = client.detection(rgb)
boxes = response['boxes']

# 4. 对每个检测到的物体进行注册和跟踪
for box in boxes:
    label = box['label']
    bbox = [int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])]
    
    # 注册（第一帧）
    response = client.register(K, rgb, depth, label=label, box=bbox)
    pose = response['pose']
    
    # 跟踪（后续帧）
    for frame in frames:
        response = client.track(K, frame_rgb, frame_depth)
        pose = response['pose']

# 5. 关闭连接
client.close()
```

---

## 服务器启动

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mesh_file` | str | (必填) | 3D模型文件路径(OBJ格式) |
| `--zmq_port` | int | 5555 | ZeroMQ服务端口 |
| `--detector_id` | str | IDEA-Research/grounding-dino-tiny | Grounding DINO模型ID |
| `--segmenter_id` | str | facebook/sam-vit-base | SAM模型ID |
| `--detection_model` | str | None | LocateAnything模型路径(可选) |
| `--debug_dir` | str | ./debug | 调试输出目录 |

### 启动命令

```bash
python zerith/zerith_server_main.py \
    --mesh_file "assets/model.obj" \
    --detection_model "/path/to/LocateAnything-3B" \
    --debug_dir "runtime/debug_zerith_server"
```

---

## 客户端启动

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mesh_file` | str | (必填) | 3D模型文件路径 |
| `--server_addr` | str | tcp://localhost:5555 | 服务器地址 |
| `--debug_dir` | str | ./client_debug | 调试输出目录 |
| `--rgb_path` | str | assets/zerith_rgb.png | RGB图像路径 |
| `--depth_path` | str | assets/zerith_depth.npy | 深度图像路径 |

### 启动命令

```bash
python zerith/zerith_client.py \
    --mesh_file "assets/model.obj" \
    --rgb_path "data/rgb.png" \
    --depth_path "data/depth.npy" \
    --server_addr "tcp://localhost:5555"
```

---

## 目录结构

### 服务器输出 (`--debug_dir`)

```
debug_dir/
├── rgb/              # 原始RGB图像
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── mask_overlay/     # 绘制了mask的图像
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── ob_in_cam/        # 物体位姿矩阵
    ├── 0.txt
    ├── 1.txt
    └── ...
```

### 客户端输出 (`--debug_dir`)

```
debug_dir/
├── label_0/
│   ├── ob_in_cam/    # 每个label单独的位姿输出
│   └── track_vis/    # 每个label单独的可视化结果
├── label_1/
│   ├── ob_in_cam/
│   └── track_vis/
└── ...
```

---

## 错误处理

所有接口响应均包含 `status` 字段：

- `status == 'success'`: 操作成功，响应包含业务数据
- `status == 'error'`: 操作失败，响应包含 `message` 字段描述错误原因

### 常见错误

| 错误消息 | 原因 | 解决方案 |
|----------|------|----------|
| Server connection failed | 服务器未启动或网络不可达 | 检查服务器状态和端口 |
| Detection failed | 检测模型未加载或推理错误 | 确保启动时指定 `--detection_model` |
| No boxes detected | 图像中未检测到目标物体 | 检查图像质量或调整检测阈值 |
| Segmentation failed | 分割模型未生成有效mask | 检查输入图像格式 |
| Registration failed | 位姿估计失败 | 检查深度图质量 |
