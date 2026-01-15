import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    HAS_TORCHVISION_RAFT = True
except ImportError:
    HAS_TORCHVISION_RAFT = False

class RAFTTracker:
    """
    RAFTTracker: 用于解决 DragGAN 在低纹理区域点漂移问题的光流追踪器。
    这是 Option 1 的核心组件，利用光流保持点在图像物体表面的一致性。
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.is_ready = False
        print("RAFT Tracker Interface initialized.")

    def load_model(self):
        """
        加载 torchvision 内置的 RAFT 预训练模型。
        """
        if not HAS_TORCHVISION_RAFT:
            print("Warning: torchvision.models.optical_flow not found. Please upgrade torchvision.")
            return

        try:
            # 使用 torchvision 提供的轻量级 RAFT 模型
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=False).to(self.device)
            self.model.eval()
            self.is_ready = True
            print("RAFT (small) model loaded successfully.")
        except Exception as e:
            print(f"Error loading RAFT model: {e}")
            self.is_ready = False

    def update_points(self, prev_img, cur_img, prev_points):
        """
        根据前后两帧图像更新点的位置。
        prev_img: numpy.ndarray (H, W, 3), 范围 [0, 255]
        cur_img: numpy.ndarray (H, W, 3), 范围 [0, 255]
        prev_points: list of [y, x]
        returns: new_points: list of [y, x]
        """
        if not self.is_ready or self.model is None:
            return prev_points

        try:
            with torch.no_grad():
                # 1. 预处理图像 (归一化并转换为 Tensor)
                # RAFT 期望输入为 (N, 3, H, W)，范围 [-1, 1]
                t_img1 = torch.from_numpy(prev_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                t_img2 = torch.from_numpy(cur_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                
                # 缩放到 [-1, 1]
                t_img1 = (t_img1 / 127.5) - 1.0
                t_img2 = (t_img2 / 127.5) - 1.0

                # 2. 调用模型计算光流
                # RAFT 的输出是一个列表，最后一项是最终的光流预测
                list_of_flows = self.model(t_img1, t_img2)
                flow = list_of_flows[-1] # (1, 2, H, W) -> [dx, dy]

                # 3. 在 prev_points 位置采样光流值
                new_points = []
                H, W = prev_img.shape[:2]
                
                for y, x in prev_points:
                    # 采样光流值 (注意坐标顺序：flow[0, 0] 是 dx, flow[0, 1] 是 dy)
                    # 我们需要将坐标转换为 grid_sample 要求的归一化范围 [-1, 1]
                    # grid_sample 的坐标是 (x, y) 顺序
                    grid_y = (y / (H - 1)) * 2 - 1
                    grid_x = (x / (W - 1)) * 2 - 1
                    grid = torch.tensor([[[[grid_x, grid_y]]]], device=self.device).float()
                    
                    # 采样
                    sampled_flow = F.grid_sample(flow, grid, align_corners=True) # (1, 2, 1, 1)
                    dx = sampled_flow[0, 0, 0, 0].item()
                    dy = sampled_flow[0, 1, 0, 0].item()
                    
                    # 4. 更新坐标
                    new_points.append([y + dy, x + dx])
                
                return new_points
        except Exception as e:
            print(f"Error in RAFT update_points: {e}")
            return prev_points
