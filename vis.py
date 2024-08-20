import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    depth_map_img = Image.open('depth_map.tiff')
    depth_map_np = np.array(depth_map_img, dtype=np.float32)
    depth_map = torch.from_numpy(depth_map_np)
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    depth_map_normalized_np = depth_map_normalized.squeeze().cpu().numpy()
    heatmap = plt.get_cmap('jet')(depth_map_normalized_np)  # 使用 'jet' colormap
    heatmap = torch.from_numpy((heatmap[:, :, :3] * 255).astype(np.uint8))  # 去掉alpha通道，转为RGB格式

    plt.imshow(heatmap)
    plt.axis('off')
    plt.show()

    heatmap_img = Image.fromarray(heatmap.numpy())
    heatmap_img.save('heatmap.png')
