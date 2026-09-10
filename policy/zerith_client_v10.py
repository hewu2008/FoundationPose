# encoding:utf8
"""Zerith client v10: reusable per-round detection and completion reporting."""

import os
import sys

import cv2

_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from zerith.zerith_client import *  # noqa: F401,F403 - keep the v9 public API
from zerith.zerith_client import ZerithFoundationPoseClient as _V9Client


class ZerithFoundationPoseClient(_V9Client):
    """v9-compatible client used by the repeated v10 perception loop."""


def create_client(server_addr):
    client = ZerithFoundationPoseClient(server_addr)
    response = client.ping()
    if response.get("status") != "success":
        print(f"Server connection failed: {response.get('message', 'Unknown error')}")
        client.close()
        return None
    print("Server connection established successfully")
    return client


def detect_parts(client, rgb_path):
    """Return (color, boxes, error); an empty boxes list is a valid result."""
    color_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color_bgr is None:
        return None, None, f"无法读取 RGB 图片: {rgb_path}"

    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    response = client.detection(color)
    if response.get("status") != "success":
        return None, None, response.get("message", "Unknown detection error")

    boxes = response.get("boxes", [])
    print(f"Detection successful, found {len(boxes)} boxes")
    return color, boxes, None
