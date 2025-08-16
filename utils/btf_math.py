# file: utils/btf_math.py
# -----------------------------------------------------------
# All tensors are torch.float32 and created on the *CPU* by
# default.  If you move inputs to CUDA (e.g. x.cuda()) the
# returned tensors follow that device automatically.
# -----------------------------------------------------------
import math
from typing import Tuple

import torch

def _safe_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / torch.clamp(v.norm(dim=-1, keepdim=True), min=eps)

# ----------  Direction  <->  Spherical coordinates  --------
def sph_to_dir(theta: torch.Tensor,
               phi:   torch.Tensor) -> torch.Tensor:
    """
    Convert semi-spherical coordinates to unit direction vectors.

    Parameters
    ----------
    theta : (N,) float32   polar angle in *radians*, 0 .. π/2
    phi   : (N,) float32   azimuth angle in *radians*, 0 .. 2π

    Returns
    -------
    dir   : (N, 3) float32   unit vectors (x, y, z)
    """
    sin_t = torch.sin(theta)
    return torch.stack((sin_t * torch.cos(phi),
                        sin_t * torch.sin(phi),
                        torch.cos(theta)), dim=-1)


def dir_to_sph(d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert unit direction vectors to semi-spherical coordinates.

    Parameters
    ----------
    d : (N, 3) float32   unit vectors

    Returns
    -------
    theta : (N,) float32   polar angle 0 .. π/2 (radians)
    phi   : (N,) float32   azimuth angle 0 .. 2π (radians)
    """
    theta = torch.acos(torch.clamp(d[..., 2], 0.0, 1.0))
    phi   = torch.atan2(d[..., 1], d[..., 0]) % (2.0 * math.pi)
    return theta, phi

# ----------  Half-vector / Difference-vector  ---------------
def half_diff(wi: torch.Tensor,
              wo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute half vector h and difference vector d.

    Parameters
    ----------
    wi, wo : (N, 3) float32   unit incident / outgoing directions

    Returns
    -------
    h, d   : (N, 3) float32   unit half / difference vectors
    """
    h = torch.nn.functional.normalize(wi + wo, dim=-1)
    d = torch.nn.functional.normalize(wi - wo, dim=-1)
    return h, d

# ----------  Rusinkiewicz 4-D encoding ----------------------
# def encode_rusinkiewicz(wi: torch.Tensor,
#                         wo: torch.Tensor) -> torch.Tensor:
#     """
#     Encode a direction pair (wi, wo) as the 4 Rusinkiewicz angles,
#     each normalised to [0, 1].
#
#         [θ_h, φ_h, θ_d, φ_d]  →  [0..1]^4
#
#     Returns
#     -------
#     feat : (N, 4) float32
#     """
#     h, d = half_diff(wi, wo)
#     th, ph = dir_to_sph(h)
#     td, pd = dir_to_sph(d)
#     return torch.stack((th * 2.0 / math.pi,         # θ_h  → [0,1]
#                         ph / (2.0 * math.pi),       # φ_h
#                         td * 2.0 / math.pi,         # θ_d
#                         pd / (2.0 * math.pi)), dim=-1)

def encode_rusinkiewicz(wi: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    """
    Proper Rusinkiewicz reparameterisation:
      - θ_h, φ_h from h = normalize(wi + wo)
      - Build an ONB around h
      - Rotate wi into that frame to measure (θ_d, φ_d)
    Returns normalised [θ_h, φ_h, θ_d, φ_d] in [0,1]^4.
    """
    # Half vector
    h = _safe_normalize(wi + wo)

    # θ_h, φ_h in the world frame
    th, ph = dir_to_sph(h)

    # Build an orthonormal basis (t_h, b_h, h) to avoid explicit rotation matrices
    # Choose an arbitrary "up" that isn’t collinear with h
    up = torch.tensor([0.0, 0.0, 1.0], device=wi.device, dtype=wi.dtype).expand_as(h)
    alt = torch.tensor([1.0, 0.0, 0.0], device=wi.device, dtype=wi.dtype).expand_as(h)
    tmp = torch.where((h[..., 2:3].abs() > 0.999), alt, up)        # (N,3)

    t_h = _safe_normalize(torch.cross(tmp, h, dim=-1))             # tangent
    b_h = torch.cross(h, t_h, dim=-1)                              # bitangent (already unit)

    # Express wi in this local frame
    wi_x = (wi * t_h).sum(dim=-1)
    wi_y = (wi * b_h).sum(dim=-1)
    wi_z = (wi * h  ).sum(dim=-1)
    wi_local = torch.stack((wi_x, wi_y, wi_z), dim=-1)

    # θ_d, φ_d measured after aligning h to the north pole
    td, pd = dir_to_sph(wi_local)

    # Normalise to [0,1]
    feat = torch.stack((th * 2.0 / math.pi,
                        ph / (2.0 * math.pi),
                        td * 2.0 / math.pi,
                        pd / (2.0 * math.pi)), dim=-1)
    return feat

# ----------  Bilinear lookup on a feature plane -------------
def bilerp(tex: torch.Tensor,
           uv:  torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation on a 2-D feature plane with optional
    circular wrapping along the u-axis (azimuth).

    Parameters
    ----------
    tex : (H, W, C) float32   feature plane
    uv  : (N, 2)   float32   normalized coords in [0, 1]^2

    Returns
    -------
    feat : (N, C) float32
    """
    H, W, C = tex.shape
    u = (uv[:, 0] % 1.0) * (W - 1)   # wrap azimuth
    v =  uv[:, 1] * (H - 1)

    x0 = torch.floor(u).long()
    x1 = (x0 + 1) % W                # wrap around W
    y0 = torch.floor(v).long()
    y1 = torch.clamp(y0 + 1, 0, H - 1)

    u_ratio = (u - x0.float()).unsqueeze(-1)  # (N,1)
    v_ratio = (v - y0.float()).unsqueeze(-1)

    p00 = tex[y0, x0]
    p10 = tex[y0, x1]
    p01 = tex[y1, x0]
    p11 = tex[y1, x1]

    return ((1 - u_ratio) * (1 - v_ratio) * p00 +
            (    u_ratio) * (1 - v_ratio) * p10 +
            (1 - u_ratio) * (    v_ratio) * p01 +
            (    u_ratio) * (    v_ratio) * p11)
