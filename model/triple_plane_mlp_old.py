# file: model/triple_plane_mlp.py
from typing import Tuple
import torch
import torch.nn as nn
from utils.btf_math import bilerp
import json
import numpy as np
from pathlib import Path
import torch.fft as fft

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
        # update max value and scale
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
        # 'quantize' path: symmetric per-tensor INT8 with STE
        s = torch.clamp(self.scale, min=self.eps)
        x_div = x / s
        x_q = torch.clamp(torch.round(x_div), -127, 127)
        # Dequant with straight-through estimator
        return (x_q - x_div).detach() * s + x

class WeightFakeQuant(nn.Module):
    """
    Per-tensor symmetric INT8 fake-quant for weights.
    Recomputes scale per forward (common QAT practice).
    """
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
    """
    Bias-free Linear with:
      - activation fake-quant on the INPUT (SymmetricFakeQuant)
      - weight fake-quant for W (WeightFakeQuant)
    """
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
        x = self.aq(x)                 # quantize activations (input to this layer)
        w = self.wq(self.linear.weight)  # quantize weights
        return torch.matmul(x, w.t())


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
                 hidden : int = 32
                 ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.u_plane = nn.Parameter(0.01 * torch.randn(u_res, u_res, channels))
        self.h_plane = nn.Parameter(0.01 * torch.randn(h_res, h_res, ang_ch))
        self.d_plane = nn.Parameter(0.01 * torch.randn(d_res, d_res, ang_ch))

        # ---- ACF synthesis state (disabled by default) ----
        self.register_buffer("_acf_enabled", torch.tensor(False), persistent=False)
        self.register_buffer("_acf_offsets", torch.zeros(0, 2), persistent=False)  # (K,2) offsets in UV cycles
        self._acf_k: int = 0
        self._acf_last_hash: int = 0  # cheap guard to avoid needless recompute

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
        # U-plane: either single fetch or ACF-blended multi-fetch
        if bool(self._acf_enabled) and self._acf_offsets.numel() > 0:
            # (B,1,2) + (1,K,2)  →  (B,K,2), wrap in [0,1]
            uvk = (uv[:, None, :] + self._acf_offsets[None, :, :]) % 1.0
            B, K, _ = uvk.shape
            uvk = uvk.reshape(B * K, 2)
            fUk = self._bilerp_linear(self.u_plane, uvk).reshape(B, K, -1)
            fU = fUk.mean(dim=1)  # uniform blend (weights sum to 1)
        else:
            fU = self._bilerp_linear(self.u_plane, uv)
        fH = bilerp(self.h_plane, h_uv)
        fD = bilerp(self.d_plane, d_uv)
        feat = torch.cat([fU, fH, fD], dim=1)
        return self.mlp(feat)

    # -----------------------
    # QAT control utilities
    # -----------------------
    def set_quant_mode(self, mode: str):
        """
        Set QAT mode on all QLinear layers.
          - 'off'      : no fake-quant anywhere
          - 'observe'  : collect activation stats; weights not quantized
          - 'quantize' : activations + weights quantize-dequantize
        """
        for m in self.mlp:
            if isinstance(m, QLinear):
                m.set_quant_mode(mode)

    @torch.no_grad()
    def export_qparams(self) -> dict:
        """
        Export current activation and weight scales per layer for deployment.
        Also provides S_a_out[L] = S_a_in[L+1] to make the re-quant chain explicit.
        """

        qls = [m for m in self.mlp if isinstance(m, QLinear)]
        a_in = {}
        a_out = {}
        w_sc = {}
        for i, ql in enumerate(qls):
            s_a_in = float(torch.clamp(ql.aq.scale, min=1e-8))
            s_w    = float(torch.clamp(torch.amax(torch.abs(ql.linear.weight)), min=1e-8) / 127.0)
            a_in[f"fc{i}"] = s_a_in
            w_sc[f"fc{i}"] = s_w
        # S_a_out[L] = S_a_in[L+1]; last has no next
        for i in range(len(qls)-1):
            a_out[f"fc{i}"] = a_in[f"fc{i+1}"]
        return {"a_scales_in": a_in, "a_scales_out": a_out, "w_scales": w_sc}
    @staticmethod
    def _pack_int8_to_i32_le(Wq: np.ndarray) -> np.ndarray:
        out_dim, in_dim = Wq.shape
        pad = (-in_dim) % 4

        if pad:
            Wq = np.pad(Wq, ((0, 0), (0, pad)), mode="constant")
        u = Wq.astype(np.int8).astype(np.uint8).astype(np.uint32)
        b0 = u[:, 0::4]
        b1 = u[:, 1::4]
        b2 = u[:, 2::4]
        b3 = u[:, 3::4]
        packed = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).astype(np.int32)
        return packed

    @torch.no_grad()
    def export_int8_package(self, out_dir, leaky_slope: float = 0.2) -> Path:
        """
        Write Int8 weights (plus dp4a-packed), planes, and a JSON manifest.
        Files:
          - manifest: qtp_manifest.json
          - weights:  fc{i}_w_int8.npy (row-major, int8)
                      fc{i}_w_i32.npy  (out_dim, ceil(in_dim/4)) int32  [dp4a little-endian packing]
          - planes :  u_plane.npy / h_plane.npy / d_plane.npy (float32)
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        qls = [m for m in self.mlp if isinstance(m, QLinear)]
        meta = {
            "version": 1,
            "target": {"cpu": "x86_64-little", "gpu": "NVIDIA-dp4a-compatible"},
            "rounding_policy": "torch.round (ties-to-even)",
            "clamp_range": [-127, 127],
            "quant": {"scheme": "symmetric per-tensor INT8"},
            "activation_encoding": "Rusinkiewicz; φ wrapped on H/D u-axis",
            "mlp": {"hidden": int(self.hidden), "qlayers": 4, "leaky_slope": float(leaky_slope)},
            "planes": {
                "u": {"shape": list(self.u_plane.shape)},
                "h": {"shape": list(self.h_plane.shape)},
                "d": {"shape": list(self.d_plane.shape)},
            },
            "u_plane_synthesis": {
            "mode": "acf_blend" if bool(self._acf_enabled) and self._acf_k > 0 else "single",
            "acf_k": int(self._acf_k)
            },
            "packing": {
                "dp4a": True,
                "endianness": "little",
                "word": "b0 | (b1<<8) | (b2<<16) | (b3<<24)"
            },
            "layers": []
        }
        # collect scales
        q = self.export_qparams()
        a_in = q["a_scales_in"]
        a_out = q["a_scales_out"]

        w_sc = q["w_scales"]

        for i, ql in enumerate(qls):
            W = ql.linear.weight.detach().cpu().float()
            s_w = w_sc[f"fc{i}"]
            # symmetric Int8 quantization of weights
            Wq = torch.clamp(torch.round(W / s_w), -127, 127).to(torch.int8).numpy()
            np.save(out / f"fc{i}_w_int8.npy", Wq, allow_pickle=False)
            np.save(out / f"fc{i}_w_i32.npy", self._pack_int8_to_i32_le(Wq), allow_pickle=False)
            layer = {
                "name": f"fc{i}",
                "in_dim": int(W.shape[1]),
                "out_dim": int(W.shape[0]),
                "S_a_in": float(a_in[f"fc{i}"]),
                "S_w": float(s_w),
            }
            if f"fc{i}" in a_out:
                layer["S_a_out"] = float(a_out[f"fc{i}"])
            meta["layers"].append(layer)

        # planes (float32)
        np.save(out / "u_plane.npy", self.u_plane.detach().cpu().numpy(), allow_pickle=False)
        np.save(out / "h_plane.npy", self.h_plane.detach().cpu().numpy(), allow_pickle=False)
        np.save(out / "d_plane.npy", self.d_plane.detach().cpu().numpy(), allow_pickle=False)

        # ============================================================
        # ACF EXPORT (tile-local, dual tiling A/B)
        # ------------------------------------------------------------
        # - We split the U-plane into square tiles (Th x Tw).
        # - For each tile, compute a local ACF pdf, take top-K offsets,
        #   and assign them to all texels in that tile.
        # - Do this twice:
        #     * Grid A: tiles start at (0,0)
        #     * Grid B: tiles start at (Th/2, Tw/2) (half-tile shift)
        # - Offsets Δuv are in normalized UV:
        #     Δu = Δx / W, Δv = Δy / H   (W,H are full texture dims)
        # Quantize Δuv ∈ [-0.5, 0.5] using full int16 SNORM range:
        # encode: q = round(clamp(Δuv, -0.5, 0.5) * 2 * 32767)
        # decode: Δuv = (q / 32767.0) * 0.5  = q / (2 * 32767.0)
        # - Scores are raw ACF magnitudes (pdf) in [0,1], float16.
        # - Shapes:
        #     acf_offsets_A.npy : (H, W, K, 2) int16
        #     acf_offsets_B.npy : (H, W, K, 2) int16
        #     acf_scores.npy    : (H, W, K)    float16   (from grid A)
        # ============================================================

        H, W, C = self.u_plane.shape
        K = int(max(1, self._acf_k))

        # --- choose a square tile size that divides H and W (fallbacks) ---
        candidates = [128, 64, 32, 16, 8, 4]
        Th = next((t for t in candidates if (H % t) == 0), H)
        Tw = next((t for t in candidates if (W % t) == 0), W)

        # Safety: cap K by tile pixel count minus the center (which we zeroed)
        max_k = max(1, min(K, Th * Tw - 1))
        if max_k != K:
            K = max_k  # we’ll also record K in manifest

        # grayscale proxy of U-plane (H,W)
        g = self.u_plane.detach().mean(dim=2)

        # allocate outputs in float first; quantize at the end
        offs_A_f = torch.empty((H, W, K, 2), dtype=torch.float32, device=g.device)
        offs_B_f = torch.empty((H, W, K, 2), dtype=torch.float32, device=g.device)
        scr_A_f  = torch.empty((H, W, K),     dtype=torch.float32, device=g.device)

        def _extract_tile(src: torch.Tensor, y0: int, x0: int, h: int, w: int) -> torch.Tensor:
            """Wrap-around extract (torus). Returns (h,w)."""
            yi = (torch.arange(h, device=src.device) + y0) % H
            xi = (torch.arange(w, device=src.device) + x0) % W
            return src.index_select(0, yi)[:, xi]

        def _write_tile(dst_offs: torch.Tensor, dst_scr: torch.Tensor,
                        y0: int, x0: int,
                        tile_offs: torch.Tensor,  # (K,2) in normalized UV
                        tile_scr: torch.Tensor):  # (K,)
            """Write a full tile region (wrap) with the same K offsets & scores."""
            yi = (torch.arange(Th, device=dst_offs.device) + y0) % H
            xi = (torch.arange(Tw, device=dst_offs.device) + x0) % W
            # expand to tile shape
            offs_tile = tile_offs.view(1, 1, K, 2).expand(Th, Tw, K, 2)
            scr_tile  = tile_scr.view(1, 1, K).expand(Th, Tw, K)
            # assign (advanced indexing with 2D index grids)
            dst_offs[yi[:, None], xi[None, :], :, :] = offs_tile
            if dst_scr is not None:
                dst_scr[yi[:, None],  xi[None, :],  :] = scr_tile

        def _compute_tile_topk(tile: torch.Tensor) -> tuple:
            """
            tile: (Th, Tw) float32
            Returns:
              - offs_uv: (K,2) float32 normalized UV (Δu=Δx/W, Δv=Δy/H)
              - scores : (K,)  float32 in [0,1]
            """
            pdf = self._acf2d_equalized(tile)  # center zeroed, sum=1
            flat = pdf.reshape(-1)
            vals, idx = torch.topk(flat, k=K, largest=True, sorted=True)
            iy = (idx // Tw).to(torch.int64)
            ix = (idx %  Tw).to(torch.int64)
            # dx,dy in texels (centered)
            dx = (ix - (Tw // 2)).float()
            dy = (iy - (Th // 2)).float()
            # normalized UV (relative to FULL texture dims)
            du = dx / float(W)
            dv = dy / float(H)
            offs_uv = torch.stack([du, dv], dim=1)  # (K,2)
            return offs_uv, vals

        # ----- Grid A (origin (0,0)) -----
        for y0 in range(0, H, Th):
            for x0 in range(0, W, Tw):
                tile = _extract_tile(g, y0, x0, Th, Tw)  # (Th,Tw)
                offs_uv, scores = _compute_tile_topk(tile)  # (K,2), (K,)
                _write_tile(offs_A_f, scr_A_f, y0, x0, offs_uv, scores)

        # ----- Grid B (half-tile shift) -----
        y_shift = (Th // 2) % H
        x_shift = (Tw // 2) % W
        for y0 in range(0, H, Th):
            for x0 in range(0, W, Tw):
                ys = (y0 + y_shift) % H
                xs = (x0 + x_shift) % W
                tile = _extract_tile(g, ys, xs, Th, Tw)
                offs_uv, _ = _compute_tile_topk(tile)  # we reuse A's scores if desired
                _write_tile(offs_B_f, None, ys, xs, offs_uv, torch.zeros(K, device=g.device))
        # quantize offsets to int16 SNORM
        # q = round(clamp(Δuv, -1, 1) * 32767); decode with Δuv = q / 32767.0
        SNORM = 32767.0
        offs_A_q = torch.clamp(offs_A_f, -0.5, 0.5).mul(2.0 * SNORM).round().to(torch.int16).cpu().numpy()
        offs_B_q = torch.clamp(offs_B_f, -0.5, 0.5).mul(2.0 * SNORM).round().to(torch.int16).cpu().numpy()
        scores_f16 = scr_A_f.to(torch.float16).cpu().numpy()

        np.save(out / "acf_offsets_A.npy", offs_A_q, allow_pickle=False)
        np.save(out / "acf_offsets_B.npy", offs_B_q, allow_pickle=False)
        np.save(out / "acf_scores.npy",    scores_f16, allow_pickle=False)

        meta["acf_export"] = {
            "K": int(K),
            "tile": {"height": int(Th), "width": int(Tw)},
            "dual_tiling": {
                "grid": "square",
                "B_anchor_shift_texels": [int(Th // 2), int(Tw // 2)]
            },
            "offsets": {
                "files": {"A": "acf_offsets_A.npy", "B": "acf_offsets_B.npy"},
                "dtype": "int16_snorm",
                "uv_norm": "Δu=Δx/W, Δv=Δy/H (normalized UV, toroidal)",
                "range_float_before_q": [-0.5, 0.5],
                "quant": {"encode": "q = round(clamp(Δuv, -0.5, 0.5) * 2 * 32767)",
                          "decode": "Δuv = q / 32767.0 * 0.5"},
                "shape_HWK2": [int(H), int(W), int(K), 2]
            },
            "scores": {
                "file": "acf_scores.npy",
                "from_grid": "A",
                "dtype": "float16",
                "range": [0.0, 1.0],
                "shape_HWK": [int(H), int(W), int(K)]
            },
            "note": "Apply user Bezier curve to scores at runtime if desired; same scores may be reused for B."
        }
        with open(out / "qtp_manifest.json", "w") as f:
            json.dump(meta, f, indent=2)

        return out

    @torch.no_grad()
    def _features_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert raw 6-D input into MLP features [fU, fH, fD] on the model's device.
        Returns shape (B, channels + 2*ang_ch).
        """
        dev = self.u_plane.device
        x = x.to(dev, non_blocking=True).float()
        uv, h_uv, d_uv = self._split_uv_hd(x)

        # U-plane fetch (with optional ACF synthesis)
        if bool(self._acf_enabled) and self._acf_offsets.numel() > 0:
            uvk = (uv[:, None, :] + self._acf_offsets[None, :, :]) % 1.0
            B, K, _ = uvk.shape
            uvk = uvk.reshape(B * K, 2)
            fUk = self._bilerp_linear(self.u_plane, uvk).reshape(B, K, -1)
            fU = fUk.mean(dim=1)
        else:
            fU = self._bilerp_linear(self.u_plane, uv)

        fH = bilerp(self.h_plane, h_uv)
        fD = bilerp(self.d_plane, d_uv)
        return torch.cat([fU, fH, fD], dim=1)

    @torch.no_grad()
    def int8_reference_forward(self, x: torch.Tensor, leaky_slope: float = 0.2) -> torch.Tensor:
        """
        CPU/torch reference of integer-only pipeline:
          qA int8  *  qW int8  →  acc int32  →  dequant float
          apply LeakyReLU (float)  →  next layer (requant at its input)
        Uses current model weights and exported scales.
        """
        device0 = x.device
        in_feat_dim = self.mlp[0].linear.in_features
        # 1) Ensure we feed the MLP with features
        if x.shape[-1] == 6:
            feat = self._features_from_input(x)  # on model device
        elif x.shape[-1] == in_feat_dim:
            feat = x.to(self.u_plane.device).float()  # already features
        else:
            raise RuntimeError(
                f"int8_reference_forward expected 6-D raw input or {in_feat_dim}-D features, got {x.shape[-1]}"
            )

        # Move to CPU for the integer math path
        y = feat.detach().cpu().float()

        # 2) INT8 pipeline for the MLP only
        qls = [m for m in self.mlp if isinstance(m, QLinear)]
        q = self.export_qparams()
        a_in = q["a_scales_in"]
        w_sc = q["w_scales"]
        for i, ql in enumerate(qls):
            s_a_in = float(a_in[f"fc{i}"])
            s_w    = float(w_sc[f"fc{i}"])
            # quantize activations and weights to int8
            Xq = torch.clamp(torch.round(y / s_a_in), -127, 127).to(torch.int8)
            W  = ql.linear.weight.detach().cpu().float()
            Wq = torch.clamp(torch.round(W / s_w), -127, 127).to(torch.int8)
            # int8 x int8 -> int32 GEMM
            acc = Xq.to(torch.int32) @ Wq.t().to(torch.int32)  # (B, out)
            y = acc.to(torch.float32) * (s_a_in * s_w)         # dequant
            if i < len(qls) - 1:
                y = torch.nn.functional.leaky_relu(y, negative_slope=leaky_slope)
        return y.to(device0)


    # -----------------------
    # ACF utilities
    # -----------------------
    @staticmethod
    def _acf2d_equalized(tex2d: torch.Tensor) -> torch.Tensor:
        """
        FFT-based unbiased autocovariance (centered), clamped to nonnegative and normalized to sum=1.
        tex2d: (H,W) float32
        returns pdf over offsets (H,W) with center zeroed to avoid trivial shift.
        """
        x = tex2d - tex2d.mean()
        F = fft.fft2(x)
        S = (F.conj() * F).real
        ac = fft.ifft2(S).real
        ac = fft.fftshift(ac)
        # remove central spike to encourage non-zero offsets
        H, W = ac.shape
        ac[H // 2, W // 2] = 0.0
        ac = torch.clamp(ac, min=0.0)
        s = ac.sum()
        if s <= 0:
            # fallback to uniform ring around center
            ac[:] = 0.0
            ac[H // 2 - 1:H // 2 + 2, W // 2] = 1.0
            ac[H // 2, W // 2 - 1:W // 2 + 2] = 1.0
            s = ac.sum()
        return ac / s

    @torch.no_grad()
    def _acf_pdf_from_uplane(self, downsample: int = 128, device: torch.device = None) -> torch.Tensor:
        """
        Build a centered, nonnegative, sum=1 ACF pdf from the U-plane mean.
        Returns a torch.Tensor on `device` (defaults to u_plane.device).
        """
        U = self.u_plane.detach()  # (H, W, C)
        g = U.mean(dim=2)  # (H, W)
        H, W = g.shape
        dsH = min(downsample, H)
        dsW = min(downsample, W)
        g_ds = torch.nn.functional.interpolate(
            g.unsqueeze(0).unsqueeze(0),
            size=(dsH, dsW), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)  # (dsH, dsW)
        pdf = self._acf2d_equalized(g_ds)  # (dsH, dsW), sum=1, center zeroed
        if device is None:
            device = self.u_plane.device
        return pdf.to(device, non_blocking=True)

    @torch.no_grad()
    def _sample_offsets_from_pdf(self, pdf: torch.Tensor, k: int) -> torch.Tensor:
        """
        pdf: (H,W), sum=1. Returns (k,2) UV-cycle offsets in [-0.5, 0.5).
        """
        H, W = pdf.shape
        flat = pdf.reshape(-1)
        idx = torch.multinomial(flat, num_samples=k, replacement=True)
        iy = (idx // W).to(torch.int64)
        ix = (idx %  W).to(torch.int64)
        # map grid indices to centered offsets
        du = (ix - (W // 2)).float() / float(W)
        dv = (iy - (H // 2)).float() / float(H)
        return torch.stack([du, dv], dim=1)

    @torch.no_grad()
    def update_acf_from_uplane(self, k: int = 4, downsample: int = 128) -> None:
        """
        Recompute ACF-based offset set from current U-plane (mean over channels),
        using the shared pdf helper.
        """
        self._acf_k = int(max(0, k))
        if self._acf_k <= 0:
            self._acf_enabled[...] = False
            self._acf_offsets = torch.zeros(0, 2, device=self.u_plane.device)
            return

        # cheap guard to avoid needless recompute if shape unchanged
        h = (self.u_plane.shape[0] << 16) ^ (self.u_plane.shape[1] << 8) ^ self.u_plane.shape[2]
        if h == self._acf_last_hash and bool(self._acf_enabled) and self._acf_offsets.numel() > 0:
            return
        self._acf_last_hash = h

        # 1) get ACF pdf (downsampled, centered, normalized)
        pdf = self._acf_pdf_from_uplane(downsample=downsample, device=self.u_plane.device)  # (Hd, Wd)

        # 2) sample K offsets from pdf (your current multinomial policy)
        offs = self._sample_offsets_from_pdf(pdf, self._acf_k).to(self.u_plane.device)  # (K, 2) cycles

        # 3) commit
        self._acf_offsets = offs
        self._acf_enabled[...] = True

    @torch.no_grad()
    def enable_acf_synthesis(self, k: int = 4, downsample: int = 128) -> None:
        self.update_acf_from_uplane(k=k, downsample=downsample)

    @torch.no_grad()
    def disable_acf_synthesis(self) -> None:
        self._acf_enabled[...] = False
        self._acf_k = 0
        self._acf_offsets = torch.zeros(0, 2, device=self.u_plane.device)

