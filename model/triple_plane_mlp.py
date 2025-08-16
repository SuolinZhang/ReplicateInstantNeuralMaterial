from pyexpat import features
from typing import Tuple
import torch
import torch.nn as nn
from utils.btf_math import bilerp
import math

class TriplePlaneMLP(nn.Module):
    """
    Input  : x (B, 6) = [u, v, θ_h, φ_h, θ_d, φ_d], all in [0,1]
    Output : RGB (B, 3), linear [0,1]
    """

    def __init__(self,
                 u_res : int = 400,
                 channels : int = 8, # u_ch : int = 16,
                 h_res : int = 50,
                 d_res : int = 50,
                 ang_ch : int = 8,
                 hidden : int = 128
                 ) -> None:
        super().__init__()
        self.u_plane = nn.Parameter(0.01 * torch.randn(u_res, u_res, channels))
        self.h_plane = nn.Parameter(0.01 * torch.randn(h_res, h_res, ang_ch))
        self.d_plane = nn.Parameter(0.01 * torch.randn(d_res, d_res, ang_ch))

        self.mlp = nn.Sequential(
            nn.Linear( channels + 2 * ang_ch, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 3)
        )
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     # --- init MLP weights for LeakyReLU(a=0.2) ---
    #     for m in self.mlp:
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity="leaky_relu")
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #
    #     # --- init feature planes (H, W, C) ---
    #     # Treat C as fan_in; small He-uniform keeps early outputs well-scaled.
    #     with torch.no_grad():
    #         fan = self.u_plane.shape[-1]
    #         bound = math.sqrt(6.0 / float(fan))
    #         self.u_plane.uniform_(-bound, bound)
    #         self.h_plane.uniform_(-bound, bound)
    #         self.d_plane.uniform_(-bound, bound)


    # @staticmethod
    # def _split_uv_hd(x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     uv = x[:, 0:2]
    #     h_uv = x[:, 2:4]
    #     d_uv = x[:, 4:6]
    #     return uv, h_uv, d_uv
    #

    @staticmethod
    def _split_uv_hd(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split inputs and reorder angular coords so that:
          - H/D planes receive (u=φ, v=θ), because bilerp wraps the u/φ axis.
        """

        uv = x[:, 0:2]  # (u, v)
        th, ph = x[:, 2], x[:, 3]  # θ_h, φ_h
        td, pd = x[:, 4], x[:, 5]  # θ_d, φ_d
        h_uv = torch.stack((ph, th), dim=1)  # (φ_h, θ_h)
        d_uv = torch.stack((pd, td), dim=1)  # (φ_d, θ_d)

        return uv, h_uv, d_uv

    def get_param_groups(self):
        """Return (plane_params_list, mlp_params_list) for separate LRs."""
        planes = []
        # Adjust names to your actual buffer names
        if hasattr(self, "u_plane"): planes.append(self.u_plane)
        if hasattr(self, "h_plane"): planes.append(self.h_plane)
        if hasattr(self, "d_plane"): planes.append(self.d_plane)

        plane_params = [p for p in planes if isinstance(p, torch.nn.Parameter)]

        mlp_params = list(self.mlp.parameters())

        return plane_params, mlp_params

    @staticmethod
    def _bilerp_linear(tex: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """
        Bilinear interpolation *without* u-axis wrapping (for the U-plane).
        tex: (H,W,C), uv: (B,2) in [0,1]
        """
        H, W, C = tex.shape
        u = torch.clamp(uv[:, 0], 0.0, 1.0) * (W - 1)
        v = torch.clamp(uv[:, 1], 0.0, 1.0) * (H - 1)

        x0 = torch.floor(u).long();
        x1 = torch.clamp(x0 + 1, 0, W - 1)
        y0 = torch.floor(v).long();
        y1 = torch.clamp(y0 + 1, 0, H - 1)

        u_ratio = (u - x0.float()).unsqueeze(-1)  # (B,1)
        v_ratio = (v - y0.float()).unsqueeze(-1)

        p00 = tex[y0, x0]
        p10 = tex[y0, x1]
        p01 = tex[y1, x0]
        p11 = tex[y1, x1]

        return ((1 - u_ratio) * (1 - v_ratio) * p00 +
                (u_ratio) * (1 - v_ratio) * p10 +
                (1 - u_ratio) * (v_ratio) * p01 +
                (u_ratio) * (v_ratio) * p11)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        uv, h_uv, d_uv = self._split_uv_hd(x)
        fU = self._bilerp_linear(self.u_plane, uv)
        fH = bilerp(self.h_plane, h_uv)
        fD = bilerp(self.d_plane, d_uv)
        feat = torch.cat([fU, fH, fD], dim=1)
        return self.mlp(feat)

