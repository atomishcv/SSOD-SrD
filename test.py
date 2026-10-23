import torch
from itertools import islice

path = "outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth"
ckpt = torch.load(path, map_location="cpu")


