import metrics
import torch
from functools import partial
import torch.nn.functional as F
from quant import *
import lasio
import numpy as np
from PIL import Image

x = torch.randn(1, 3, 128, 128)
b, c, h, w = x.shape
patch_size = 32
padding = patch_size // 4
stride = patch_size // 2
x = F.pad(x, (padding, padding, padding, padding))
x = x.unfold(2, patch_size, stride)
print(x.shape)
x = x.unfold(3, patch_size, stride)
print(x.shape)
