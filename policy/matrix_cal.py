import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def main():
    T2 = np.eye(4)
    T2[:3, :3] = R.from_euler('xyz', [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
    T2[:3, 3] = [0.2194, 0.0325, 0.6075]

    T3 = np.eye(4)
    T3[:3, 3] = [-0.5743, -0.1800, -0.1208]

    print(np.dot(T3, T2))

if __name__ == "__main__":
    main()