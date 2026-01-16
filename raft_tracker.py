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
                    py, px = y + dy, x + dx

                    # 4. 自适应混合微调 (Adaptive Hybrid Refinement)
                    if feat_resize is not None and feat_refs is not None and r2 is not None:
                        # 计算局部特征显著性 (Feature Salience)
                        r_s = 2 # 显著性检测窗口
                        u_s, d_s = max(int(py-r_s), 0), min(int(py+r_s+1), H)
                        l_s, r_s_idx = max(int(px-r_s), 0), min(int(px+r_s+1), W)
                        local_feat = feat_resize[0, :, u_s:d_s, l_s:r_s_idx]
                        
                        # 计算局部标准差作为纹理丰富度的度量
                        feat_std = torch.std(local_feat, dim=(1, 2)).mean().item()
                        
                        # 只有当区域具有足够的特征显著性时，才进行特征匹配微调
                        # 阈值 0.05 是一个经验值，用于区分纯色皮毛和有纹理区域
                        if feat_std > 0.05:
                            r = max(3, round(r2 / 512 * H * 0.4))
                            up = max(int(round(py - r)), 0)
                            down = min(int(round(py + r + 1)), H)
                            left = max(int(round(px - r)), 0)
                            right = min(int(round(px + r + 1)), W)
                            
                            feat_patch = feat_resize[:, :, up:down, left:right]
                            L2 = torch.linalg.norm(feat_patch - feat_refs[i].reshape(1, -1, 1, 1), dim=1)
                            _, idx = torch.min(L2.view(1, -1), -1)
                            width = right - left
                            py = idx.item() // width + up
                            px = idx.item() % width + left
                        else:
                            # 在纯色区域，我们更信任光流，并且限制点的随机漂移
                            # 如果提供了 targets，当接近目标时进一步锁定
                            if targets is not None:
                                ty, tx = targets[i]
                                dist_to_target = np.sqrt((py - ty)**2 + (px - tx)**2)
                                if dist_to_target < 3.0:
                                    # 接近目标时，引入强阻尼，防止飘走
                                    py = py * 0.8 + ty * 0.2
                                    px = px * 0.8 + tx * 0.2

                    # 边界检查
                    ny = max(0, min(py, H - 1))
                    nx = max(0, min(px, W - 1))
                    new_points.append([ny, nx])
                
                return new_points
        except Exception as e:
            print(f"Error in RAFT update_points: {e}")
            return prev_points
