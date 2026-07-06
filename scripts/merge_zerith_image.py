# encoding:utf8

import os
import shutil
import re

SRC_DIR = '/home/jszn/tmp/'
DST_DIR = '/home/jszn/hewu/dataset/locate_anything/'
START_NUM = 63


def get_png_files(directory):
    png_files = []
    for f in os.listdir(directory):
        if f.lower().endswith('.png'):
            match = re.match(r'^(\d+)\.png$', f)
            if match:
                png_files.append((int(match.group(1)), f))
    png_files.sort(key=lambda x: x[0])
    return [f for _, f in png_files]


def main():
    src_files = get_png_files(SRC_DIR)
    print(f"Found {len(src_files)} PNG files in {SRC_DIR}")

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    current_num = START_NUM
    for src_file in src_files:
        src_path = os.path.join(SRC_DIR, src_file)
        dst_path = os.path.join(DST_DIR, f"{current_num}.png")
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_file} -> {current_num}.png")
        current_num += 1

    print(f"\nDone! Copied {len(src_files)} files, last file: {current_num - 1}.png")


if __name__ == '__main__':
    main()
