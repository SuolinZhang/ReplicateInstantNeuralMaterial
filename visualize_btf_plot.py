import matplotlib.pyplot as plt
import numpy as np
from btf_extractor import Ubo2014

btf = Ubo2014("data/leather11_W400xH400_L151xV151.btf")
ti, pi, to, po = next(iter(btf.angles_set))      # first of 22 801 pairs
img = btf.angles_to_image(ti, pi, to, po)      # (400,400,3) uint8
img_rgb = img[..., ::-1]
img_srgb = np.clip(img_rgb ** (1/2.2), 0, 1)

plt.imshow(img_srgb)
plt.title(f"L=({ti:.1f},{pi:.1f})  V=({to:.1f},{po:.1f})")
plt.axis("off")
plt.show()
