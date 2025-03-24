import torch
import math
from time import time

from modules.utils import *


class RayMarcher(torch.nn.Module):
    def __init__(self, batch=1, layer=4, precision="fp32", mode="2minus"):
        super().__init__()
        self.B = batch
        self.L = layer
        self.dtype = torch.float16 if precision == "fp16" else torch.float32

        H = W = 512
        cx = W / 2.0
        cy = H / 2.0
        f = W / 1.0
        scalar = 1.0

        y = torch.arange(H, dtype=self.dtype, device="cuda").unsqueeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, W, 1, 1) - cy
        x = torch.arange(W, dtype=self.dtype, device="cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(H, 1, 1, 1) - cx
        self.pixel_coord = torch.empty(H, W, self.B, self.L, 4, dtype=self.dtype, device="cuda")
            
        self.pixel_coord[..., 0] = -y / f * scalar
        self.pixel_coord[..., 1] = x / f * scalar
        self.pixel_coord[..., 2] = -f / f
        self.pixel_coord[..., 3] = 1.0
        self.pixel_coord = self.pixel_coord.unsqueeze(-1)  # (H, W, B, L, 4, 1)
        
        self.density = torch.empty(self.B, self.L, H, W, dtype=self.dtype, device="cuda")
        self.image = torch.empty(self.B, 3, H, W, dtype=self.dtype, device="cuda")
        
        self.signs = torch.tensor([
            [ 1,  1,  1],
            [-1, -1,  1],
            [ 1, -1, -1],
            [-1,  1, -1],
        ], dtype=self.dtype, device="cuda")  # (L, 3)

    def forward(self, sh_tensor, inv_tf):

        B, N, H, W = sh_tensor.shape

        assert N % self.L == 0
        assert B == self.B
        
        sh_tensor = sh_tensor.view(self.B, self.L, 19, H, W)

        layered_inv_tf = inv_tf.unsqueeze(0).expand(-1, self.L, -1, -1).clone()  # (B, L, 3, 4)
        layered_inv_tf[:, :, :, 3] *= self.signs.unsqueeze(0)

        world_coord = torch.matmul(layered_inv_tf, self.pixel_coord).squeeze(dim=-1)  # (H, W, B, L, 3)
        dirs = world_coord / torch.norm(world_coord, dim=-1, keepdim=True)
        sh_basis = eval_sh(dirs, 16)  # (H, W, B, L, 16)
        basis = sh_basis.permute(2, 3, 4, 0, 1)  # (B, L, 16, H, W)
        
        self.density = torch.sum(sh_tensor[:, :, :16] * basis, dim=2)  # (B, L, H, W)
        
        density = torch.softmax(self.density, dim=1)
        
        self.image[:, 0] = torch.sum(density * sh_tensor[:, :, 16], dim=1)
        self.image[:, 1] = torch.sum(density * sh_tensor[:, :, 17], dim=1)
        self.image[:, 2] = torch.sum(density * sh_tensor[:, :, 18], dim=1)

        image = torch.clamp(self.image, -1.0, 1.0)

        # features = torch.cat((image, density, r_layer, g_layer, b_layer), dim=1)

        return image, None  # B 3 H W
