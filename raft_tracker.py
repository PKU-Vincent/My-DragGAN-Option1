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

    def update_points(self, prev_img, cur_img, prev_points, fb_check=True, 
                      feat_resize=None, feat_refs=None, r2=None, targets=None):
        """
        根据前后两帧图像更新点的位置。
        prev_img: numpy.ndarray (H, W, 3), 范围 [0, 255]
        cur_img: numpy.ndarray (H, W, 3), 范围 [0, 255]
        prev_points: list of [y, x]
        fb_check: 是否启用前后向一致性检查
        feat_resize: 可选, 当前帧的 GAN 特征图 (1, C, H, W)
        feat_refs: 可选, 初始帧的参考特征点列表
        r2: 搜索半径
        targets: 可选, 目标点列表 [[y, x], ...]
        returns: new_points: list of [y, x]
        """
        if not self.is_ready or self.model is None:
            return prev_points

        try:
            with torch.no_grad():
                # 1. 预处理图像
                t_img1 = torch.from_numpy(prev_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                t_img2 = torch.from_numpy(cur_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                
                t_img1 = (t_img1 / 127.5) - 1.0
                t_img2 = (t_img2 / 127.5) - 1.0

                # 2. 计算光流
                list_of_flows_fwd = self.model(t_img1, t_img2)
                flow_fwd = list_of_flows_fwd[-1]

                if fb_check:
                    list_of_flows_bwd = self.model(t_img2, t_img1)
                    flow_bwd = list_of_flows_bwd[-1]

                # 3. 更新点位置
                new_points = []
                H, W = prev_img.shape[:2]
                
                for i, (y, x) in enumerate(prev_points):
                    grid_y = (y / (H - 1)) * 2 - 1
                    grid_x = (x / (W - 1)) * 2 - 1
                    grid = torch.tensor([[[[grid_x, grid_y]]]], device=self.device).float()
                    
                    s_flow_fwd = F.grid_sample(flow_fwd, grid, align_corners=True)
                    dx = s_flow_fwd[0, 0, 0, 0].item()
                    dy = s_flow_fwd[0, 1, 0, 0].item()
                    
                    if fb_check:
                        y2, x2 = y + dy, x + dx
                        grid_y2 = (y2 / (H - 1)) * 2 - 1
                        grid_x2 = (x2 / (W - 1)) * 2 - 1
                        grid2 = torch.tensor([[[[grid_x2, grid_y2]]]], device=self.device).float()
                        
                        s_flow_bwd = F.grid_sample(flow_bwd, grid2, align_corners=True)
                        dx_back = s_flow_bwd[0, 0, 0, 0].item()
                        dy_back = s_flow_bwd[0, 1, 0, 0].item()
                        
                        err = np.sqrt((dx + dx_back)**2 + (dy + dy_back)**2)
                        weight = np.exp(-(err**2) / (2.0 * 1.0**2))
                        weight = max(0.1, weight)
                        
                        dx *= weight
                        dy *= weight

                    # RAFT 初步预测的位置
                    py_raft, px_raft = y + dy, x + dx

                    # 4. 自适应代价函数追踪 (Adaptive Cost-Function Tracking)
                    if feat_resize is not None and feat_refs is not None and r2 is not None:
                        # 恢复较大的搜索半径以保证速度 (Speed-First)
                        # 我们不再限制半径，而是通过代价函数来约束漂移
                        r = round(r2 / 512 * H)
                        up = max(int(round(py_raft - r)), 0)
                        down = min(int(round(py_raft + r + 1)), H)
                        left = max(int(round(px_raft - r)), 0)
                        right = min(int(round(px_raft + r + 1)), W)
                        
                        feat_patch = feat_resize[:, :, up:down, left:right]
                        # 1) 特征相似度代价 (L2 Distance)
                        L2 = torch.linalg.norm(feat_patch - feat_refs[i].reshape(1, -1, 1, 1), dim=1) # (1, h, w)
                        
                        # 2) 光流一致性惩罚 (RAFT Distance Penalty)
                        # 创建网格坐标
                        patch_h, patch_w = L2.shape[1], L2.shape[2]
                        yy, xx = torch.meshgrid(
                            torch.arange(up, down, device=self.device),
                            torch.arange(left, right, device=self.device),
                            indexing='ij'
                        )
                        dist_to_raft = torch.sqrt((yy - py_raft)**2 + (xx - px_raft)**2).unsqueeze(0)
                        
                        # 3) 目标引导偏置 (Target Bias) - 针对平坦区域加速
                        target_bias = torch.zeros_like(dist_to_raft)
                        if targets is not None:
                            ty, tx = targets[i]
                            dist_to_target = torch.sqrt((yy - ty)**2 + (xx - tx)**2).unsqueeze(0)
                            # 距离目标越近，代价越小 (引导点往目标跑)
                            target_bias = dist_to_target * 0.1 

                        # 计算局部特征显著性
                        local_feat = feat_resize[0, :, up:down, left:right]
                        feat_std = torch.std(local_feat, dim=(1, 2)).mean().item()
                        
                        # 动态调整权重：
                        # 在纹理丰富区 (feat_std高)，信任特征匹配；
                        # 在平坦区 (feat_std低)，信任光流和目标引导。
                        lambda_raft = 0.5 if feat_std > 0.05 else 2.0
                        
                        # 总代价函数
                        cost = L2 + lambda_raft * dist_to_raft + target_bias
                        
                        _, idx = torch.min(cost.view(1, -1), -1)
                        py = idx.item() // patch_w + up
                        px = idx.item() % patch_w + left
                        
                        # 目标锁定 (到达后强行稳住)
                        if targets is not None:
                            ty, tx = targets[i]
                            dist_final = np.sqrt((py - ty)**2 + (px - tx)**2)
                            if dist_final < 2.0:
                                py, px = ty, tx
                    else:
                        py, px = py_raft, px_raft

                    # 边界检查
                    ny = max(0, min(py, H - 1))
                    nx = max(0, min(px, W - 1))
                    new_points.append([ny, nx])
                
                return new_points
        except Exception as e:
            print(f"Error in RAFT update_points: {e}")
            return prev_points
