# encoding:utf8
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
depth_array = np.load("/home/jszn/hewu/alg-product/FoundationPose/assets/zerith_depth.npy") 

plt.figure(figsize=(8, 6))

plt.imshow(depth_array / 1000.0, cmap='viridis') 

plt.colorbar(label='Distance (meters)')

plt.title('Depth Map Viewer')
plt.axis('off') 
plt.show()