import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import math
from math import ceil
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

def tv2d(x: torch.Tensor) -> torch.Tensor: # adjusted
    dx = x[1:, :, :] - x[:-1, :, :]
    dy = x[:, 1:, :] - x[:, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def seed_everything(seed : int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def pin_worker_to_core(worker_id: int):
    # 1) unique RNG per worker (keeps sampling independent)
    info = torch.utils.data.get_worker_info()
    seed = 42 + (worker_id if info is None else info.id)
    import random; random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)

    # 2) CPU affinity (Linux): pin each worker to a distinct core if possible
    try:
        import os
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

def psnr(pred: torch.Tensor, tgt: torch.Tensor, eps: float=1e-8) -> float:
    mse = torch.mean((pred - tgt) ** 2).item()
    if mse <=0: return 99.0
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
                accum_steps: int = 1) -> float:
    model.train()
    running = 0.0

    # for x, y in loader:
    #     x = x.to(device, non_blocking=True)
    #     y = y.to(device, non_blocking=True)
    #     pred = model(x)
    #     loss = torch.mean(torch.abs(pred - y))  # L1 loss
    #     # tv_u = tv2d(model.u_plane)
    #     # tv_a = tv2d(model.h_plane) + tv2d(model.d_plane)
    #
    #
    #     opt.zero_grad(set_to_none=True)
    #     loss.backward()
    #     # (optional) mild grad clip helps with huge batches
    #     # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #     opt.step()
    #
    #     running += loss.detach().item()

    accum_steps = max(1, int(accum_steps))
    num_batches = len(loader)
    opt.zero_grad(set_to_none=True)


    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = torch.mean(torch.abs(pred - y))  # L1 loss
        # tv_u = tv2d(model.u_plane)
        # tv_a = tv2d(model.h_plane) + tv2d(model.d_plane)

        (loss / accum_steps).backward()

        do_step = (i % accum_steps == 0) or (i == num_batches)

        if do_step:
            # (optional) mild grad clip helps with huge batches
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        running += loss.detach().item()

    return running / max(1, len(loader))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btf", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2560000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() -2)))
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--ang_channels", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--u_res", type=int, default=400)
    ap.add_argument("--h_res", type=int, default=50)
    ap.add_argument("--d_res", type=int, default=50)
    ap.add_argument("--samples", type=int, default=2560000)
    ap.add_argument("--out", type=str, default="runs/fp32")
    ap.add_argument("--val_n", type=int, default=12)
    ap.add_argument("--reuse_per_frame", type=int, default=128)
    ap.add_argument("--cache_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--accum_steps", type=int, default=10)
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out);
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    loader = make_loader(args.btf, samples=args.samples, batch=args.batch,
                         workers=args.workers, reuse_per_frame=args.reuse_per_frame,
                         cache_size=args.cache_size)

    btf = Ubo2014(args.btf)  # main-process BTF for validation/GT
    angles_list = btf.angles_set
    val_set = pick_val_angles(angles_list, n=args.val_n)

    model = TriplePlaneMLP(u_res=args.u_res, channels=args.channels,
                           h_res=args.h_res, d_res=args.d_res, ang_ch=args.ang_channels,
                           hidden=args.hidden).to(device)

    plane_params, mlp_params = model.get_param_groups()
    opt = torch.optim.AdamW([
        {"params": plane_params, "lr": 1e-3},  # planes 1e-3
        {"params": mlp_params, "lr": 3e-4},  # MLP   3e-4
    ])
    # sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)

    # opt = AdamW(model.parameters(), lr=args.lr)
    sched = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=5e-5)


    best_psnr = -1e9
    print("== FP32 training start ==")
    if args.accum_steps and args.accum_steps > 1:
        print(f"[INFO] Gradient accumulation: accum_steps={args.accum_steps}")
    for ep in range(1, args.epochs + 1):
        # loss = train_epoch(model, loader, opt, device)
        loss = train_epoch(model, loader, opt, device, accum_steps=args.accum_steps)
        sched.step()

        # lrs = [pg["lr"] for pg in opt.param_groups]
        # print(f"[EP {ep:02d}] train L1 = {loss:.5f} | lr_planes={lrs[0]:.6f} lr_mlp={lrs[1]:.6f}")
        lr_now = opt.param_groups[0]["lr"]
        print(f"[EP {ep:02d}] train L1 = {loss:.5f} | lr={lr_now:.6f}")

        avg_mae, avg_psnr = validate_epoch(
            model, btf, val_set, out_dir / f"val_ep{ep:02d}", device
        )

        # always keep per-epoch checkpoint if you like
        ckpt = out_dir / "ckpt" / f"fp32_ep{ep:02d}.pth"
        torch.save({"ep": ep, "model": model.state_dict(), "opt": opt.state_dict()}, ckpt)

        # save best-by-avg-PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_ckpt = out_dir / "ckpt" / "best.pth"
            torch.save({"ep": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                        "avg_psnr": best_psnr}, best_ckpt)
            print(f"[CKPT] New best by PSNR ({best_psnr:.2f} dB) → {best_ckpt}")

    print(f"Done. Results under: {out_dir}")

if __name__ == "__main__":
    main()



















