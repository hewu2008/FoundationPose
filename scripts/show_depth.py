# encoding:utf8
"""深度图交互查看器：鼠标移动即显示该像素的深度值与相机系坐标。

用法：
    python scripts/show_depth.py                          # 默认看 zero 参考深度
    python scripts/show_depth.py --npy <path> --scale 0.001
    python scripts/show_depth.py --npy <path> --save out.png

说明：
  * ``--scale auto``（默认）在最大值大于 20 时按毫米处理（除以 1000），
    否则按米处理；本项目 ``zerith_ref_depth.npy`` 是 float32 米。
  * 深度 <= 0 的像素视为无效（无回波/空洞），用黑色显示，
    读数处显示 ``depth=invalid``。
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.widgets import Cursor

DEFAULT_NPY = (
    '/home/jszn/hewu/alg-product/FoundationPose/assets/imgs/zerith_ref_depth.npy'
)

# Zerith 采集相机的内参，与 scripts/zerith_foundation_pose_server.py 一致。
FX, FY, CX, CY = 607.62, 608.40, 329.68, 243.36

# 读数里有中文，matplotlib 默认字体不含 CJK 字形（会显示成方块），
# 因此优先选用系统里可用的中文字体。
CJK_FONTS = (
    'Noto Sans CJK SC',
    'Noto Sans CJK JP',
    'WenQuanYi Micro Hei',
    'Source Han Sans SC',
    'SimHei',
)


def setup_cjk_font():
    """把第一个可用的中文字体插到最前面，保证中文读数正常显示。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name] + list(
                plt.rcParams['font.sans-serif']
            )
            break
    plt.rcParams['axes.unicode_minus'] = False


def load_depth(path, scale='auto'):
    """读取深度 npy，返回 (米为单位的深度图, 实际使用的 scale)。"""
    raw = np.load(path).astype(np.float64).squeeze()
    if raw.ndim != 2:
        raise SystemExit(f'期望二维深度图，实际 shape={raw.shape}')
    if scale == 'auto':
        scale = 0.001 if np.nanmax(raw) > 20.0 else 1.0
    else:
        scale = float(scale)
    return raw * scale, scale


def readout(u, v, depth):
    """返回鼠标所在像素的读数文本（深度 + 反投影的相机系 XYZ）。"""
    height, width = depth.shape
    if not (0 <= u < width and 0 <= v < height):
        return '超出图像范围'
    z = depth[v, u]
    if not np.isfinite(z) or z <= 0:
        return f'u={u}  v={v}   depth=invalid (无回波/空洞)'
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return (
        f'u={u}  v={v}   depth={z:.4f} m ({z * 1000:.1f} mm)   '
        f'XYZ_cam=({x * 1000:.1f}, {y * 1000:.1f}, {z * 1000:.1f}) mm'
    )


def main():
    parser = argparse.ArgumentParser(description='深度图查看器（鼠标读取深度值）')
    parser.add_argument('--npy', default=DEFAULT_NPY, help='深度 .npy 路径')
    parser.add_argument(
        '--scale',
        default='auto',
        help='深度缩放：auto（默认，>20 视为毫米）或具体倍数',
    )
    parser.add_argument('--title', default=None, help='窗口标题，默认用文件名')
    parser.add_argument('--save', default=None, help='可选：额外保存一张预览图')
    args = parser.parse_args()

    depth, scale = load_depth(args.npy, args.scale)
    setup_cjk_font()
    valid = depth > 0
    values = np.where(valid, depth, np.nan)

    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(top=0.90, bottom=0.08)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('black')  # 无效像素显示为黑色，与有效回波区分
    image = ax.imshow(values, cmap=cmap)
    ax.set_title(args.title or f'{args.npy}   (scale={scale})')
    ax.axis('off')
    fig.colorbar(image, ax=ax, label='Depth (m)', fraction=0.046, pad=0.02)

    text = fig.text(0.02, 0.955, '把鼠标移到深度图上读取该点数值', va='top')
    Cursor(ax, useblit=True, color='red', linewidth=1)

    def show_value(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        u = int(event.xdata + 0.5)
        v = int(event.ydata + 0.5)
        text.set_text(readout(u, v, depth))
        fig.canvas.draw_idle()

    def print_value(event):
        # 点击时把读数同时打印到终端，方便复制。
        if event.inaxes is ax and event.xdata is not None:
            print(readout(int(event.xdata + 0.5), int(event.ydata + 0.5), depth))

    fig.canvas.mpl_connect('motion_notify_event', show_value)
    fig.canvas.mpl_connect('button_press_event', print_value)

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f'preview saved: {args.save}')
    plt.show()


if __name__ == '__main__':
    main()
