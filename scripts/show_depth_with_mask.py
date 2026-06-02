# encoding:utf8
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import Cursor

# 读取深度图和mask
depth_array = np.load("/home/jszn/hewu/alg-product/FoundationPose/debug_limiter/erode_depth.npy") 
mask = cv2.imread("/home/jszn/hewu/alg-product/FoundationPose/debug_limiter/ob_mask.png", cv2.IMREAD_GRAYSCALE)

# 创建图形
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. 原始深度图
ax1 = axes[0]
im1 = ax1.imshow(depth_array / 1000.0, cmap='viridis')
ax1.set_title('Original Depth Map')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, label='Distance (meters)')

# 2. Mask图像
ax2 = axes[1]
im2 = ax2.imshow(mask, cmap='gray')
ax2.set_title('Object Mask')
ax2.axis('off')
plt.colorbar(im2, ax=ax2)

# 3. Mask叠加到深度图
ax3 = axes[2]
depth_normalized = depth_array / 1000.0
depth_colored = plt.cm.viridis((depth_normalized - depth_normalized.min()) / 
                               (depth_normalized.max() - depth_normalized.min()))
depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)

mask_colored = np.zeros_like(depth_colored)
mask_colored[mask > 0] = [255, 0, 0]

alpha = 0.5
overlay = depth_colored.copy()
overlay[mask > 0] = (depth_colored[mask > 0] * (1 - alpha) + mask_colored[mask > 0] * alpha).astype(np.uint8)

ax3.imshow(overlay)
ax3.set_title('Depth Map with Mask Overlay')
ax3.axis('off')

plt.tight_layout()

# 添加像素值显示功能
# 创建一个文本框显示像素值
text = fig.text(0.02, 0.98, '', transform=fig.transFigure, va='top')

# 存储各子图的数据
images_data = {
    'depth': depth_array / 1000.0,
    'mask': mask,
    'overlay': overlay
}

axes_data = {
    ax1: ('Depth', images_data['depth']),
    ax2: ('Mask', images_data['mask']),
    ax3: ('Overlay', images_data['overlay'])
}

def update_pixel_info(event):
    """当鼠标移动时更新像素值显示"""
    for ax in axes:
        if ax == event.inaxes:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            name, data = axes_data[ax]
            
            if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                if data.ndim == 2:
                    # 灰度图像
                    value = data[y, x]
                    text_str = f'图: {name} | 位置: ({x}, {y}) | 值: {value:.4f}'
                else:
                    # RGB图像
                    value = data[y, x]
                    text_str = f'图: {name} | 位置: ({x}, {y}) | RGB: ({value[0]}, {value[1]}, {value[2]})'
            else:
                text_str = '图: 超出范围'
            
            text.set_text(text_str)
            break

# 绑定鼠标移动事件
fig.canvas.mpl_connect('motion_notify_event', update_pixel_info)

# 添加十字光标
for ax in axes:
    Cursor(ax, useblit=True, color='red', linewidth=1)

plt.savefig('/home/jszn/hewu/alg-product/FoundationPose/debug_limiter/depth_with_mask.png', dpi=150)
plt.show()

print(f"Saved to: /home/jszn/hewu/alg-product/FoundationPose/debug_limiter/depth_with_mask.png")
