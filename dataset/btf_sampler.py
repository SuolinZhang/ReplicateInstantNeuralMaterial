"""
file: dataset/btf_sampler
Dynamic BTF → sample-pair generator
===================================

  Returns (x, y) pairs that a network can consume directly.

    x = [u, v, θ_h, φ_h, θ_d, φ_d]   (float32, all ∈ [0,1])
    y = [R, G, B]                    (float32, linear, 0-1)

•   **Dynamic sampling** – nothing is pre-baked to disk.
•   Compatible with Python 3.8, torch 2.2, btf-extractor.
"""

from typing import Optional, Union, Tuple
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from btf_extractor import Ubo2014
from utils.btf_math import sph_to_dir, encode_rusinkiewicz

# btf_sampler.py  (add params, lazy open, reuse, cache)
class BTFSampler(Dataset):

    def __init__(self, btf_path: Union[str, Path], n_samples: int,
                 seed: Optional[int] = 42,
                 reuse_per_frame: int = 128,
                 cache_size: int = 16) -> None:
        super().__init__()
        self.btf_path = str(btf_path)
        self.n_samples = int(n_samples)
        self.reuse_per_frame = int(reuse_per_frame)
        self.cache_size = int(cache_size)

        self.rng = random.Random(int(seed) if seed is not None else 42)

        # Lazy-open state (created inside worker)
        self.btf = None
        self.angles = None
        self.H = self.W = None

        # Reuse/cache state (per worker)
        self._reuses_left = 0
        self._current_img = None  # numpy RGB float32 [0,1], (H,W,3)
        self._current_key = None  # (ti,pi,to,po)
        self._cache = {}  # key -> np.ndarray
        self._lru = []  # LRU keys

    def __len__(self):
        # how many (x,y) training pairs you want the loader to draw in total
        return int(self.n_samples)

    def _ensure_open(self):
        if self.btf is None:
            self.btf = Ubo2014(self.btf_path)
            self.angles = np.asarray(list(self.btf.angles_set), dtype=np.float32)
            H, W, _ = self.btf.img_shape
            self.H, self.W = int(H), int(W)

    def _decode_rgb01(self, ti, pi, to, po):
        bgr = self.btf.angles_to_image(float(ti), float(pi), float(to), float(po))
        if bgr.dtype == np.uint8:
            return np.ascontiguousarray(bgr[..., ::-1]).astype(np.float32) / 255.0
        return np.ascontiguousarray(bgr[..., ::-1]).astype(np.float32)

    def _get_frame(self, key):
        if key in self._cache:
            # move to MRU
            try: self._lru.remove(key)
            except ValueError: pass
            self._lru.append(key)
            return self._cache[key]
        img = self._decode_rgb01(*key)
        self._cache[key] = img
        self._lru.append(key)
        if len(self._lru) > self.cache_size:
            old = self._lru.pop(0)
            self._cache.pop(old, None)
        return img

    def __getitem__(self, idx: int):
        self._ensure_open()

        # Reuse the same decoded image for multiple samples
        if self._reuses_left <= 0 or self._current_img is None:
            ti, pi, to, po = self.angles[self.rng.randrange(len(self.angles))]
            key = (float(ti), float(pi), float(to), float(po))
            self._current_img = self._get_frame(key)
            self._current_key = key
            self._reuses_left = self.reuse_per_frame

        # random UV → pixel
        u = self.rng.random(); v = self.rng.random()
        px = int(u * (self.W - 1)); py = int(v * (self.H - 1))
        rgb = torch.from_numpy(self._current_img[py, px].copy()).to(torch.float32)  # (3,)

        # encode directions for current angle (Rusinkiewicz already in your utils)
        ti, pi, to, po = self._current_key
        wi = sph_to_dir(torch.tensor(np.radians(ti), dtype=torch.float32),
                        torch.tensor(np.radians(pi), dtype=torch.float32))
        wo = sph_to_dir(torch.tensor(np.radians(to), dtype=torch.float32),
                        torch.tensor(np.radians(po), dtype=torch.float32))
        dir_feat = encode_rusinkiewicz(wi.unsqueeze(0), wo.unsqueeze(0)).squeeze(0)

        x = torch.cat([torch.tensor([u, v], dtype=torch.float32), dir_feat])  # (6,)
        y = rgb
        self._reuses_left -= 1
        return x, y