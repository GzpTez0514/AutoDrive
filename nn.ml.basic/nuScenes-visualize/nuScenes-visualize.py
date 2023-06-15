# pip install nuscenes-devkit
from nuscenes.nuscenes import NuScenes
import numpy as np
import cv2
import os

# 构建nuscenes类
version = 'v1.0-mini'
dataroot = '/data/nuscenes/nuscenes_mini'
nuscenes = NuScenes(version, dataroot, verbose=False) 
print(len(nuscenes.sample))

sample = nuscenes.sample[0]
print(sample)

# 获取lidar的数据
