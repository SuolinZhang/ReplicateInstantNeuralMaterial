# file: model/triple_plane_mlp.py
from typing import Tuple
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn

from utils.btf_math import bilerp


# -------------------------
# Quantization primitives
# -------------------------
class SymmetricFakeQuant(nn.Module):
    """
    Per-tensor symmetric INT8 fake-quant for activations with EMA calibration.
    Modes:
      - 'off'               : pass-through
      - 'observe'           : update running amax/scale, pass-through
      - 'quantize_observe'  : quantize-dequantize (STE) AND keep updating scale
      - 'quantize'          : quantize-dequantize (STE) with frozen scale
    """
    def __init__(self, ema_decay: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("running_amax", torch.zeros(()))
        self.register_buffer("scale", torch.ones(()))
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.mode = "off"

    @torch.no_grad()
    def _observe(self, x: torch.Tensor):
        amax = torch.amax(torch.abs(x))
        if torch.isfinite(amax):
            d = self.ema_decay
            self.running_amax.mul_(d).add_(amax * (1.0 - d))
            self.scale.copy_(torch.clamp(self.running_amax, min=self.eps) / 127.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return x
        if self.mode == "observe":
            self._observe(x)
            return x
        if self.mode == "quantize_observe":
            self._observe(x)
        s = torch.clamp(self.scale, min=self.eps)
        x_div = x / s
        x_q = torch.clamp(torch.round(x_div), -127, 127)
        return (x_q - x_div).detach() * s + x


class WeightFakeQuant(nn.Module):
    """Per-tensor symmetric INT8 fake-quant for weights (recomputed every fwd)."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.enabled = False
        self.eps = float(eps)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return w
        s = torch.clamp(torch.amax(torch.abs(w)), min=self.eps) / 127.0
        w_div = w / s
        w_q = torch.clamp(torch.round(w_div), -127, 127)
        return (w_q - w_div).detach() * s + w


class QLinear(nn.Module):
    """Bias-free Linear with activation + weight fake-quant."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.aq = SymmetricFakeQuant()
        self.wq = WeightFakeQuant()

    def set_quant_mode(self, mode: str):
        assert mode in ("off", "observe", "quantize", "quantize_observe")
        self.aq.mode = mode
        self.wq.enabled = (mode in ("quantize", "quantize_observe"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aq(x)
        w = self.wq(self.linear.weight)
        return torch.matmul(x, w.t())


# -------------------------
# Triple-plane MLP (clean)
# -------------------------
class TriplePlaneMLP(nn.Module):
    """
    Input  : x (B,6) = [u, v, θ_h, φ_h, θ_d, φ_d] in [0,1]
    Output : (B,3) RGB (linear [0,1])
    Notes:
      - H/D planes expect (u=θ, v=φ) to match shader/CUDA ordering:
        tex2DLayered(HP, h1, h2, …) where (h1,h2) = (θ, φ).
      - U-plane is sampled with plain bilinear; runtime synthesis (T/InvT + hashed UV)
        is applied in Falcor/CUDA and is not trained through.
    """
    def __init__(
        self,
        u_res: int = 400,
        channels: int = 8,
        h_res: int = 50,
        d_res: int = 50,
        ang_ch: int = 8,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.u_plane = nn.Parameter(0.01 * torch.randn(u_res, u_res, channels))
        self.h_plane = nn.Parameter(0.01 * torch.randn(h_res, h_res, ang_ch))
        self.d_plane = nn.Parameter(0.01 * torch.randn(d_res, d_res, ang_ch))

        in_ch = channels + 2 * ang_ch
        self.mlp = nn.Sequential(
            QLinear(in_ch, hidden),
            nn.ReLU(inplace=True),
            QLinear(hidden, hidden),
            nn.ReLU(inplace=True),
            QLinear(hidden, hidden),
            nn.ReLU(inplace=True),
            QLinear(hidden, 3),
        )

    # ---------- helpers ----------
    @staticmethod
    def _split_uv_hd(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split inputs so that Python bilerp (which wraps u) sees (φ, θ).
        At export we transpose H/D so the Falcor/CUDA side can use (θ, φ).
        """
        uv = x[:, 0:2]
        th, ph = x[:, 2], x[:, 3]
        td, pd = x[:, 4], x[:, 5]
        # Correct order for runtime: (θ, φ)
        h_uv = torch.stack((ph, th), dim=1)
        d_uv = torch.stack((pd, td), dim=1)
        return uv, h_uv, d_uv

    @staticmethod
    def _bilerp_linear(tex: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation without wrapping. tex: (H,W,C), uv: (B,2) in [0,1]."""
        H, W, C = tex.shape
        u = torch.clamp(uv[:, 0], 0.0, 1.0) * (W - 1)
        v = torch.clamp(uv[:, 1], 0.0, 1.0) * (H - 1)
        x0 = torch.floor(u).long()
        x1 = torch.clamp(x0 + 1, 0, W - 1)
        y0 = torch.floor(v).long()
        y1 = torch.clamp(y0 + 1, 0, H - 1)
        u_ratio = (u - x0.float()).unsqueeze(-1)
        v_ratio = (v - y0.float()).unsqueeze(-1)
        p00 = tex[y0, x0]
        p10 = tex[y0, x1]
        p01 = tex[y1, x0]
        p11 = tex[y1, x1]
        return (
            (1 - u_ratio) * (1 - v_ratio) * p00
            + u_ratio * (1 - v_ratio) * p10
            + (1 - u_ratio) * v_ratio * p01
            + u_ratio * v_ratio * p11
        )

    def _features_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw (…,6) to concatenated features on the model device."""
        dev = self.u_plane.device
        x = x.to(dev, non_blocking=True).float()
        uv, h_uv, d_uv = self._split_uv_hd(x)
        fU = self._bilerp_linear(self.u_plane, uv)
        fH = bilerp(self.h_plane, h_uv)
        fD = bilerp(self.d_plane, d_uv)
        return torch.cat([fU, fH, fD], dim=1)

    # ---------- training / inference ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._features_from_input(x)
        return self.mlp(feat)

    def get_param_groups(self):
        """(plane_params, mlp_params) for separate optimizers/LRs."""
        planes = [p for p in (self.u_plane, self.h_plane, self.d_plane) if isinstance(p, nn.Parameter)]
        return planes, list(self.mlp.parameters())

    def set_quant_mode(self, mode: str):
        for m in self.mlp:
            if isinstance(m, QLinear):
                m.set_quant_mode(mode)

    @torch.no_grad()
    def export_qparams(self) -> dict:
        """Collect activation and weight scales per QLinear layer."""
        qls = [m for m in self.mlp if isinstance(m, QLinear)]
        a_in, a_out, w_sc = {}, {}, {}
        for i, ql in enumerate(qls):
            s_a = float(torch.clamp(ql.aq.scale, min=1e-8))
            s_w = float(torch.clamp(torch.amax(torch.abs(ql.linear.weight)), min=1e-8) / 127.0)
            a_in[f"fc{i}"] = s_a
            w_sc[f"fc{i}"] = s_w
        for i in range(len(qls) - 1):
            a_out[f"fc{i}"] = a_in[f"fc{i+1}"]
        return {"a_scales_in": a_in, "a_scales_out": a_out, "w_scales": w_sc}

    @staticmethod
    def _pack_int8_to_i32_le(Wq: np.ndarray) -> np.ndarray:
        """dp4a-friendly little-endian packing: 4x int8 → 1x int32."""
        out_dim, in_dim = Wq.shape
        pad = (-in_dim) % 4
        if pad:
            Wq = np.pad(Wq, ((0, 0), (0, pad)), mode="constant")
        u = Wq.astype(np.int8).astype(np.uint8).astype(np.uint32)
        b0 = u[:, 0::4]; b1 = u[:, 1::4]; b2 = u[:, 2::4]; b3 = u[:, 3::4]
        return (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).astype(np.int32)

    @torch.no_grad()
    def int8_reference_forward(self, x: torch.Tensor, leaky_slope: float = 0.0) -> torch.Tensor:
        """
        CPU/torch reference of the integer GEMM chain for the MLP (features in float).
        Accepts raw x(…,6) or precomputed features(…, in_dim).
        """
        device0 = x.device
        in_feat = self.mlp[0].linear.in_features
        if x.shape[-1] == 6:
            y = self._features_from_input(x).float().cpu()
        elif x.shape[-1] == in_feat:
            y = x.float().cpu()
        else:
            raise RuntimeError(f"expected 6 or {in_feat} features, got {x.shape[-1]}")
        qls = [m for m in self.mlp if isinstance(m, QLinear)]
        q = self.export_qparams(); a_in, w_sc = q["a_scales_in"], q["w_scales"]
        for i, ql in enumerate(qls):
            s_a = float(a_in[f"fc{i}"]); s_w = float(w_sc[f"fc{i}"])
            Xq = torch.clamp(torch.round(y / s_a), -127, 127).to(torch.int8)
            W  = ql.linear.weight.detach().cpu().float()
            Wq = torch.clamp(torch.round(W / s_w), -127, 127).to(torch.int8)
            acc = Xq.to(torch.int32) @ Wq.t().to(torch.int32)
            y   = acc.to(torch.float32) * (s_a * s_w)
            if i < len(qls) - 1:
                if leaky_slope == 0.0:
                    y = torch.relu(y)
                else:
                    y = torch.nn.functional.leaky_relu(y, negative_slope=leaky_slope)
        return y.to(device0)

    # ---------- exports ----------
    @torch.no_grad()
    def export_int8_package(self, out_dir, leaky_slope: float = 0.0) -> Path:
        """
        Writes:
          - fc{i}_w_int8.npy : int8 weights (row-major)
          - fc{i}_w_i32.npy  : dp4a-packed int32 weights
          - u_plane.npy / h_plane.npy / d_plane.npy
          - qtp_manifest.json (minimal)
        """
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        qls = [m for m in self.mlp if isinstance(m, QLinear)]
        meta = {
            "version": 1,
            "mlp": {"hidden": int(self.hidden), "qlayers": 4, "leaky_slope": float(leaky_slope)},
            "planes": {"u": {"shape": list(self.u_plane.shape)},
                       "h": {"shape": list(self.h_plane.shape)},
                       "d": {"shape": list(self.d_plane.shape)}},
            "layers": []
        }
        q = self.export_qparams()
        a_in, a_out, w_sc = q["a_scales_in"], q["a_scales_out"], q["w_scales"]
        for i, ql in enumerate(qls):
            W = ql.linear.weight.detach().cpu().float()
            s_w = w_sc[f"fc{i}"]
            Wq = torch.clamp(torch.round(W / s_w), -127, 127).to(torch.int8).numpy()
            np.save(out / f"fc{i}_w_int8.npy", Wq, allow_pickle=False)
            np.save(out / f"fc{i}_w_i32.npy", self._pack_int8_to_i32_le(Wq), allow_pickle=False)
            layer = {"name": f"fc{i}", "in_dim": int(W.shape[1]), "out_dim": int(W.shape[0]),
                     "S_a_in": float(a_in[f"fc{i}"]), "S_w": float(s_w)}
            if f"fc{i}" in a_out:
                layer["S_a_out"] = float(a_out[f"fc{i}"])
            meta["layers"].append(layer)
        np.save(out / "u_plane.npy", self.u_plane.detach().cpu().numpy(), allow_pickle=False)
        np.save(out / "h_plane.npy", self.h_plane.detach().cpu().numpy(), allow_pickle=False)
        np.save(out / "d_plane.npy", self.d_plane.detach().cpu().numpy(), allow_pickle=False)
        with open(out / "qtp_manifest.json", "w") as f:
            json.dump(meta, f, indent=2)
        return out

    @torch.no_grad()
    def export_falcor_package(self, out_dir: str, name: str) -> Path:
        """
        Falcor/CUDA friendly export:
          - Weight_int8_{name}.bin  : float32 file; values are INT8 weights in [-127..127],
                                      row-major per layer, each row padded so in_dim%4==0.
          - UPlane_{name}.bin       : float32, (layers,H,W,4) packing
          - HPlane_{name}.bin
          - DPlane_{name}.bin
          - PlaneMeta_{name}.bin    : [u_res, u_layers, h_res, h_layers, d_res, d_layers] (float32)
          - Scales_{name}.bin       : [8 floats] activation/weight chain
        """
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

        # --- collect per-layer quant params ---
        q = self.export_qparams()
        a_in, w_sc = q["a_scales_in"], q["w_scales"]
        qls = [m for m in self.mlp if isinstance(m, QLinear)]

        # -------------------------------------------------
        # Weights → float32 values (numerically int8), row-major
        # Each row is padded to make in_dim a multiple of 4
        # -------------------------------------------------
        weights_i8 = []
        for i, ql in enumerate(qls):
            W = ql.linear.weight.detach().cpu().float()                 # (out, in)
            s_w = float(w_sc[f"fc{i}"])
            Wq = torch.clamp(torch.round(W / s_w), -127, 127).to(torch.int8).numpy()
            out_dim, in_dim = Wq.shape
            pad = (-in_dim) % 4
            if pad:
                # pad zeros on the input dimension so groups of 4 can be dp4a-packed
                Wq = np.pad(Wq, ((0, 0), (0, pad)), mode="constant")
                in_dim += pad
            # row-major flatten: [row0 all inputs], [row1 all inputs], ...
            weights_i8.append(Wq.reshape(-1).astype(np.int8))

        w_all_i8 = np.concatenate(weights_i8, axis=0)          # int8 stream
        w_all_f32 = w_all_i8.astype(np.float32)                # Falcor reads float
        # 4-per-word guarantee after per-row padding
        assert (w_all_f32.size % 4) == 0, "total weight count must be divisible by 4"
        w_all_f32.tofile(out / f"Weight_int8_{name}.bin")

        # ------------- planes (unchanged) -------------
        def export_plane(plane: torch.Tensor, tag: str, transpose_hw: bool = False):
            arr = plane.detach().cpu().numpy().astype(np.float32)  # (H, W, C) with Python’s (φ along W, θ along H)
            if transpose_hw:
                arr = arr.transpose(1, 0, 2)  # → (W, H, C) so x=θ, y=φ for runtime

            H, W, C = arr.shape
            arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
            pad = (-C) % 4
            if pad:
                arr = np.pad(arr, ((0, pad), (0, 0), (0, 0)), "constant")
            C = arr.shape[0]
            layers = C // 4
            arr = arr.reshape(layers, 4, H, W).transpose(0, 2, 3, 1)  # (layers, H, W, 4)
            arr.tofile(out / f"{tag}_{name}.bin")
            return H, layers

        u_res, u_layers = export_plane(self.u_plane, "UPlane", transpose_hw=False)
        h_res, h_layers = export_plane(self.h_plane, "HPlane", transpose_hw=True)
        d_res, d_layers = export_plane(self.d_plane, "DPlane", transpose_hw=True)

        np.array([u_res, u_layers, h_res, h_layers, d_res, d_layers], dtype=np.float32).tofile(
            out / f"PlaneMeta_{name}.bin"
        )

        # Scales blob (kept as before)
        scales = np.array(
            [
                a_in["fc0"], a_in["fc0"] * w_sc["fc0"],
                a_in["fc1"], a_in["fc1"] * w_sc["fc1"],
                a_in["fc2"], a_in["fc2"] * w_sc["fc2"],
                a_in["fc3"], a_in["fc3"] * w_sc["fc3"],
            ],
            dtype=np.float32,
        )
        scales.tofile(out / f"Scales_{name}.bin")

        return out
