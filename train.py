# train.py
import matplotlib
matplotlib.use("Agg")          # headless save
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import math
import argparse
from pathlib import Path
from typing import Tuple, List
from btf_extractor import Ubo2014
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import imageio.v2 as imageio

from dataset.btf_sampler import BTFSampler
from model.triple_plane_mlp import TriplePlaneMLP
from utils.btf_math import sph_to_dir, encode_rusinkiewicz

# enable cudnn autotune + fast matmul
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except:
    pass


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def pin_worker_to_core(worker_id: int):
    # Unique RNG per worker (keeps sampling independent)
    info = torch.utils.data.get_worker_info()
    seed = 42 + (worker_id if info is None else info.id)
    import random; random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    # Optional CPU affinity (Linux)
    try:
        n = os.cpu_count() or 24
        core = (worker_id % n)
        os.sched_setaffinity(0, {core})
    except Exception:
        pass  # non-Linux or no permission


def make_loader(path: str, samples: int, batch: int, workers: int,
                reuse_per_frame: int = 128, cache_size: int = 16) -> DataLoader:
    ds = BTFSampler(path, n_samples=samples, seed=42,
                    reuse_per_frame=reuse_per_frame, cache_size=cache_size)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=max(1, workers),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        worker_init_fn=pin_worker_to_core,
    )


def psnr(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - tgt) ** 2).item()
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / (mse + eps))


@torch.no_grad()
def render_one_frame(model: nn.Module,
                     ti: float, pi: float, to: float, po: float,
                     H: int, W: int, device: torch.device) -> torch.Tensor:
    """Render an H×W frame (linear RGB in [0,1]) by querying over UV grid."""
    v = torch.linspace(0, 1, H, device=device)
    u = torch.linspace(0, 1, W, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')  # (H,W)
    uv = torch.stack([uu, vv], dim=-1).reshape(-1, 2)               # (N,2)

    wi = sph_to_dir(torch.tensor([math.radians(ti)], dtype=torch.float32, device=device),
                    torch.tensor([math.radians(pi)], dtype=torch.float32, device=device))  # (1,3)
    wo = sph_to_dir(torch.tensor([math.radians(to)], dtype=torch.float32, device=device),
                    torch.tensor([math.radians(po)], dtype=torch.float32, device=device))  # (1,3)

    # encode_rusinkiewicz must output [θ_h, φ_h, θ_d, φ_d] in [0,1]
    dir_feat = encode_rusinkiewicz(wi, wo).repeat(uv.shape[0], 1)   # (N,4)

    x = torch.cat([uv, dir_feat], dim=1)                            # (N,6)
    rgb = model(x).reshape(H, W, 3).clamp(0, 1).detach().cpu()
    return rgb


@torch.no_grad()
def validate_epoch(model: nn.Module,
                   dataset: Ubo2014,
                   angles_list: List[Tuple[float, float, float, float]],
                   out_dir: Path,
                   device: torch.device) -> Tuple[float, float]:
    """Save GT vs Pred PNGs and print MAE/PSNR for a fixed angle tuple."""
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W, _ = dataset.img_shape
    maes, psnrs = [], []
    for k, (ti, pi, to, po) in enumerate(angles_list):
        pred = render_one_frame(model, ti, pi, to, po, H, W, device)  # (H,W,3)

        gt_frame = dataset.angles_to_image(ti, pi, to, po)        # (H,W,3), BGR
        if gt_frame.dtype == np.uint8:
            gt = torch.from_numpy(gt_frame[..., ::-1].copy()).float() / 255.0
        else:
            gt = torch.from_numpy(gt_frame[..., ::-1].copy()).float()

        mae = torch.mean(torch.abs(pred - gt)).item()
        _psnr = psnr(pred, gt)

        maes.append(mae)
        psnrs.append(_psnr)

        view_dir = out_dir / f"view_{k:02d}_ti{ti:.0f}_pi{pi:.0f}_to{to:.0f}_po{po:.0f}"
        view_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(view_dir / "pred.png", (pred.numpy() * 255).astype(np.uint8))
        imageio.imwrite(view_dir / "gt.png",   (gt.numpy()   * 255).astype(np.uint8))

    avg_mae = float(np.mean(maes))
    avg_psnr = float(np.mean(psnrs))
    print(f"[VAL] {len(angles_list)} views | MAE={avg_mae:.5f}  PSNR={avg_psnr:.2f} dB  | saved to {out_dir}")
    return avg_mae, avg_psnr


def pick_val_angles(angles_set, n: int = 12) -> List[Tuple[float, float, float, float]]:
    """Deterministically pick ~n evenly spaced angle tuples from the dataset’s angles_set."""
    angles_list = sorted(list(angles_set))  # (ti, pi, to, po) in degrees
    if len(angles_list) == 0:
        raise ValueError("angles_set is empty.")
    n = min(n, len(angles_list))
    idxs = np.linspace(0, len(angles_list) - 1, num=n, dtype=int)
    return [tuple(map(float, angles_list[i])) for i in idxs]


def train_epoch(model: nn.Module,
                loader: DataLoader,
                opt: torch.optim.Optimizer,
                device: torch.device,
                accum_steps: int = 1,
                *,
                qat: bool = False,
                qat_calib_steps: int = 0,
                qat_freeze_after: int = 0,
                global_step: int = 0) -> tuple:
    model.train()
    running = 0.0
    accum_steps = max(1, int(accum_steps))
    num_batches = len(loader)
    opt.zero_grad(set_to_none=True)

    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = torch.mean(torch.abs(pred - y))  # L1 loss

        (loss / accum_steps).backward()

        do_step = (i % accum_steps == 0) or (i == num_batches)
        if do_step:
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            # ---- QAT: switch after calibration ----
            if qat and global_step == int(qat_calib_steps):
                if hasattr(model, "set_quant_mode"):
                    try:
                        if int(qat_freeze_after) > 0:
                            model.set_quant_mode("quantize_observe")
                            print(f"[QAT] → 'quantize_observe' at step {global_step} "
                                  f"(freeze after {int(qat_freeze_after)} steps).")
                        else:
                            model.set_quant_mode("quantize")
                            print(f"[QAT] → 'quantize' at step {global_step}.")
                    except AssertionError:
                        model.set_quant_mode("quantize")
                        print(f"[QAT] → 'quantize' at step {global_step} (no 'quantize_observe').")

            if qat and int(qat_freeze_after) > 0 and global_step == int(qat_calib_steps) + int(qat_freeze_after):
                if hasattr(model, "set_quant_mode"):
                    model.set_quant_mode("quantize")
                print(f"[QAT] Scales frozen at step {global_step} (mode='quantize').")

        running += loss.detach().item()

    return running / max(1, len(loader)), global_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btf", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2560000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() - 2)))
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--ang_channels", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--u_res", type=int, default=400)
    ap.add_argument("--h_res", type=int, default=50)
    ap.add_argument("--d_res", type=int, default=50)
    ap.add_argument("--samples", type=int, default=2560000)
    ap.add_argument("--out", type=str, default="runs/fp32")
    ap.add_argument("--val_n", type=int, default=12)
    ap.add_argument("--reuse_per_frame", type=int, default=128)
    ap.add_argument("--cache_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--accum_steps", type=int, default=2)

    # ---- QAT options ----
    ap.add_argument("--qat", action="store_true",
                    help="Enable INT8 symmetric QAT for MLP (planes stay FP32).")
    ap.add_argument("--qat_calib_steps", type=int, default=200,
                    help="Optimiser steps to observe activation stats before quantizing.")
    ap.add_argument("--qat_freeze_after", type=int, default=0,
                    help="Extra steps to keep observing while quantizing before freezing scales.")

    # ---- Export / reference gates ----
    ap.add_argument("--export_int8_dir", type=str, default=None,
                    help="If set, export Int8 weights + manifest after training.")
    ap.add_argument("--int8_ref_check", action="store_true",
                    help="Run CPU Int8 reference forward on a mini-batch and report MAE.")
    ap.add_argument("--int8_ref_mae_max", type=float, default=1e-3,
                    help="Export gate: max allowed MAE between QAT and CPU-Int8.")
    ap.add_argument("--int8_ref_psnr_min", type=float, default=50.0,
                    help="Export gate: min allowed PSNR between QAT and CPU-Int8.")
    # ---- Falcor bundle export ----
    ap.add_argument("--export_falcor_dir", type=str, default=None,
                    help="If set, export Falcor binary bundle (Weight_int8_*.bin, U/H/DPlane_*.bin, PlaneMeta_*.bin).")
    ap.add_argument("--falcor_name", type=str, default="custom_int8",
                    help="Name token <name> used in Falcor filenames: Weight_int8_<name>.bin, UPlane_<name>.bin, ...")
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    train_losses = []

    loader = make_loader(args.btf, samples=args.samples, batch=args.batch,
                         workers=args.workers, reuse_per_frame=args.reuse_per_frame,
                         cache_size=args.cache_size)

    btf = Ubo2014(args.btf)
    angles_list = btf.angles_set
    val_set = pick_val_angles(angles_list, n=args.val_n)

    model = TriplePlaneMLP(u_res=args.u_res, channels=args.channels,
                           h_res=args.h_res, d_res=args.d_res, ang_ch=args.ang_channels,
                           hidden=args.hidden).to(device)

    if args.qat and hasattr(model, "set_quant_mode"):
        model.set_quant_mode("observe")
        print(f"[QAT] Starting in 'observe' mode for {args.qat_calib_steps} steps.")

    planes, mlp = model.get_param_groups()
    opt = torch.optim.AdamW([
        {"params": planes, "lr": 1e-3, "weight_decay": 1e-6, "name": "planes"},
        {"params": mlp,    "lr": 5e-4, "weight_decay": 1e-4, "name": "mlp"},
    ])
    sched = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)

    best_psnr = -1e9
    print("== Training start ==")
    if args.accum_steps and args.accum_steps > 1:
        print(f"[INFO] Gradient accumulation: accum_steps={args.accum_steps}")

    global_step = 0
    for ep in range(1, args.epochs + 1):
        loss, global_step = train_epoch(
            model, loader, opt, device,
            accum_steps=args.accum_steps,
            qat=args.qat,
            qat_calib_steps=args.qat_calib_steps,
            qat_freeze_after=args.qat_freeze_after,
            global_step=global_step,
        )
        train_losses.append(float(loss))
        sched.step()

        lrs = [pg["lr"] for pg in opt.param_groups]
        print(f"[EP {ep:02d}] train L1 = {loss:.5f} | lr_planes={lrs[0]:.6f} lr_mlp={lrs[1]:.6f}")

        # Validate with 'quantize' mode if QAT is enabled, then restore
        prev_mode = None
        if args.qat and hasattr(model, "set_quant_mode"):
            prev_mode = "observe"
            try:
                for m in model.mlp:
                    if getattr(m, "aq", None) is not None:
                        prev_mode = m.aq.mode
                        break
            except Exception:
                pass
            model.set_quant_mode("quantize")

        avg_mae, avg_psnr = validate_epoch(
            model, btf, val_set, out_dir / f"val_ep{ep:02d}", device)

        if args.qat and hasattr(model, "set_quant_mode") and prev_mode is not None:
            model.set_quant_mode(prev_mode)

        # Save checkpoint
        ckpt = out_dir / "ckpt" / f"fp32_ep{ep:02d}.pth"
        extra_qat = {}
        if args.qat and hasattr(model, "export_qparams"):
            try:
                extra_qat = model.export_qparams()
            except Exception:
                extra_qat = {}
        torch.save({"ep": ep,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "qat": extra_qat},
                   ckpt)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_ckpt = out_dir / "ckpt" / "best.pth"
            torch.save({"ep": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                        "avg_psnr": best_psnr, "qat": extra_qat}, best_ckpt)
            print(f"[CKPT] New best by PSNR ({best_psnr:.2f} dB) → {best_ckpt}")

    # Plot training curve
    fig = plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Train L1 Loss"); plt.title("Training Loss")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    fig.savefig(out_dir / "train_loss.png", dpi=150)
    plt.close(fig)
    print(f"Done. Results under: {out_dir}")

    # ----- Optional: export Int8 package -----
    if args.export_int8_dir is not None:
        if args.qat and hasattr(model, "set_quant_mode"):
            model.set_quant_mode("quantize")
        gate_bs = min(args.batch, 65536)
        it = iter(make_loader(args.btf, samples=gate_bs, batch=gate_bs,
                              workers=args.workers, reuse_per_frame=args.reuse_per_frame,
                              cache_size=args.cache_size))
        x_ref, _ = next(it)
        x_ref = x_ref.to(device, non_blocking=True)
        with torch.no_grad():
            y_qat = model(x_ref).detach().cpu()
            y_int = model.int8_reference_forward(x_ref).detach().cpu()
        mae_gate = float(torch.mean(torch.abs(y_qat - y_int)))
        psnr_gate = psnr(y_qat, y_int)
        print(f"[INT8-REF:EXPORT] MAE={mae_gate:.6f}  PSNR={psnr_gate:.2f} dB")
        if mae_gate <= args.int8_ref_mae_max and psnr_gate >= args.int8_ref_psnr_min:
            save_dir = Path(args.export_int8_dir)
            path = model.export_int8_package(save_dir, leaky_slope=0.0)
            print(f"[EXPORT] Int8 weights + manifest written to: {path}")
        else:
            print("[EXPORT] ABORTED: parity gate failed "
                  f"(MAE>{args.int8_ref_mae_max} or PSNR<{args.int8_ref_psnr_min}).")

    # ----- Optional: CPU Int8 reference parity check -----
    if args.int8_ref_check:
        if args.qat and hasattr(model, "set_quant_mode"):
            model.set_quant_mode("quantize")
        gate_bs = min(args.batch, 65536)
        it = iter(make_loader(args.btf, samples=gate_bs, batch=gate_bs,
                              workers=args.workers, reuse_per_frame=args.reuse_per_frame,
                              cache_size=args.cache_size))
        x_ref, y_ref = next(it)
        x_ref = x_ref.to(device, non_blocking=True)
        with torch.no_grad():
            y_qat = model(x_ref).detach().cpu()
            y_int = model.int8_reference_forward(x_ref).detach().cpu()
        mae = float(torch.mean(torch.abs(y_qat - y_int)))
        print(f"[INT8-REF] MAE(QAT vs CPU-Int8) on one batch: {mae:.6f}")

    # ----- Optional: export Falcor binary bundle -----
    if args.export_falcor_dir is not None:
        # Use quantized behaviour for final export
        if args.qat and hasattr(model, "set_quant_mode"):
            model.set_quant_mode("quantize")
        path = model.export_falcor_package(args.export_falcor_dir, args.falcor_name)
        print(f"[EXPORT] Falcor bundle written to: {path}")
if __name__ == "__main__":
    main()
