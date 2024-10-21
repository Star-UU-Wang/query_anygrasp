from PIL import Image
import numpy as np
import cv2

# load RGB
rgb_image = Image.open('./example_data/color.png') 
rgb_format = rgb_image.mode  # image mode
rgb_shape = rgb_image.size + (len(rgb_image.getbands()),)  # Shape (W x H x C)

# load depth
depth_image = cv2.imread('./example_data/depth.png', cv2.IMREAD_UNCHANGED) 
depth_shape = depth_image.shape  # Shape

# output
print(f"RGB图像格式: {rgb_format}")
print(f"RGB图像形状: {rgb_shape}")
print(f"深度图像形状: {depth_shape}")
