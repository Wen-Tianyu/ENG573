from torch import nn
from quant import Quanter_N
import torch.nn.functional as F
import torch


class Interpolate(nn.Module):
    def __init__(self, scale_factor=None, mode='nearest'):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x,
                             scale_factor=self.scale_factor,
                             mode=self.mode)


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = lambda x: x
        self.measurer = nn.Sigmoid()  # nn.HardSigmoid() will break down the system
        self.quanter = Quanter_N()
        self.decoder = lambda x: x

    def encode(self, x):
        return self.encoder(x)

    def measure(self, x):
        return self.measurer(x)

    def quant(self, x):
        return self.quanter.quant(x)

    def dequant(self, x):
        return self.quanter.dequant(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        # x = self.measure(x)
        x = self.decode(x)
        return x

    @torch.no_grad()
    def compress(self, x):
        x = self.encode(x)
        x = self.measure(x)
        x = self.quant(x)
        return x

    @torch.no_grad()
    def decompress(self, x):
        x = self.dequant(x)
        x = self.decode(x)
        return x


class MLP_AE(AE):
    def __init__(self, en_dim):
        super(MLP_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 28 * 28, en_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(en_dim, 3 * 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (3, 28, 28))
        )


class TransConvAE(AE):
    def __init__(self):
        super(TransConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.Sigmoid()
        )


class InterpolateAE(AE):
    def __init__(self, gray=False):
        super(InterpolateAE, self).__init__()
        self.in_dim = 1 if gray else 3
        self.em_dim = 1 if gray else 1

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_dim, 16, 3, padding=1),  # 3*128*128 -> 16*128*128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*128*128 -> 16*64*64
            nn.Conv2d(16, self.em_dim, 3, padding=1),  # 16*64*64 -> 4*64*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4*64*64 -> 4*32*32
        )

        self.decoder = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),  # 4*32*32 -> 4*64*64
            nn.Conv2d(self.em_dim, 16, 3, padding=1),  # 4*64*64 -> 16*64*64
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*64*64 -> 16*128*128
            nn.Conv2d(16, self.in_dim, 3, padding=1),  # 16*128*128 -> 3*128*128
            nn.Sigmoid()
        )


class InterpolateAE_MS(AE):
    def __init__(self, gray=False, patch_size=32):
        super(InterpolateAE_MS, self).__init__()
        self.hc, self.wc = None, None
        self.in_dim = 1 if gray else 3
        self.em_dim = 1 if gray else 1
        self.patch_size = patch_size

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_dim, 16, 3, padding=1),  # 3*128*128 -> 16*128*128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*128*128 -> 16*64*64
            nn.Conv2d(16, self.em_dim, 3, padding=1),  # 16*64*64 -> 4*64*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4*64*64 -> 4*32*32
        )

        self.decoder = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),  # 4*32*32 -> 4*64*64
            nn.Conv2d(self.em_dim, 16, 3, padding=1),  # 4*64*64 -> 16*64*64
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*64*64 -> 16*128*128
            nn.Conv2d(16, 16, 3, padding=1),  # 16*128*128 -> 3*128*128
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, self.in_dim, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def divide(self, x):
        # x.shape = [batch_size, channel, H, W]
        b, c, h, w = x.shape
        patch_size = self.patch_size
        padding = patch_size // 4
        stride = patch_size // 2
        x = F.pad(x, (padding, padding, padding, padding))
        x = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        self.hc, self.wc = x.shape[2:4]
        x = x.reshape(b, c, -1, patch_size, patch_size).permute(0, 2, 1, 3, 4).reshape(-1, c, patch_size, patch_size)
        return x

    def merge(self, x):
        # x.shape = [batch_size * patches, channel, patch_size, patch_size]
        hc, wc = self.hc, self.wc
        b = x.shape[0] // (hc * wc)
        c = x.shape[1]
        ps = self.patch_size
        x = x.reshape(b, hc, wc, c, ps, ps)
        x = x[..., ps // 4:ps // 4 * 3, ps // 4:ps // 4 * 3]
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, hc * ps // 2, wc * ps // 2)
        return x

    def forward(self, x):
        x = self.divide(x)
        x = self.encode(x)
        # x = self.measure(x)
        x = self.decode(x)
        x = self.merge(x)
        x = self.sigmoid(self.final_conv(x))
        return x


class InterpolateAE_S(AE):
    def __init__(self, gray=False):
        super(InterpolateAE_S, self).__init__()
        self.in_dim = 1 if gray else 3
        self.em_dim = 1 if gray else 4

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_dim, 16, 3, padding=1),  # 3*128*128 -> 16*128*128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*128*128 -> 16*64*64
            nn.Conv2d(16, 16, 3, padding=1),  # 16*64*64 -> 16*64*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*64*64 -> 16*32*32
            nn.Conv2d(16, self.em_dim, 3, padding=1),  # 16*32*32 -> 4*32*32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4*32*32 -> 4*16*16
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.em_dim, 16, 3, padding=1),  # 4*16*16 -> 16*16*16
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*16*16 -> 16*32*32
            nn.Conv2d(16, 16, 3, padding=1),  # 16*32*32 -> 16*32*32
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*32*32 -> 16*64*64
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*64*64 -> 16*128*128
            nn.Conv2d(16, self.in_dim, 3, padding=1),  # 16*128*128 -> 8*128*128
            nn.Sigmoid()
        )


class InterpolateAE_S_MS(AE):
    def __init__(self, gray=False, patch_size=32):
        super(InterpolateAE_S_MS, self).__init__()
        self.hc, self.wc = None, None
        self.in_dim = 1 if gray else 3
        self.em_dim = 1 if gray else 4
        self.patch_size = patch_size

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_dim, 16, 3, padding=1),  # 3*128*128 -> 16*128*128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*128*128 -> 16*64*64
            nn.Conv2d(16, 16, 3, padding=1),  # 16*64*64 -> 16*64*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*64*64 -> 16*32*32
            nn.Conv2d(16, self.em_dim, 3, padding=1),  # 16*32*32 -> 4*32*32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4*32*32 -> 4*16*16
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.em_dim, 16, 3, padding=1),  # 4*16*16 -> 16*16*16
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*16*16 -> 16*32*32
            nn.Conv2d(16, 16, 3, padding=1),  # 16*32*32 -> 16*32*32
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*32*32 -> 16*64*64
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*64*64 -> 16*128*128
            nn.Conv2d(16, 16, 3, padding=1),  # 16*128*128 -> 8*128*128
            # nn.Sigmoid()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, self.in_dim, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def divide(self, x):
        # x.shape = [batch_size, channel, H, W]
        b, c, h, w = x.shape
        patch_size = self.patch_size
        padding = patch_size // 4
        stride = patch_size // 2
        x = F.pad(x, (padding, padding, padding, padding))
        x = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        self.hc, self.wc = x.shape[2:4]
        x = x.reshape(b, c, -1, patch_size, patch_size).permute(0, 2, 1, 3, 4).reshape(-1, c, patch_size, patch_size)
        return x

    def merge(self, x):
        # x.shape = [batch_size * patches, channel, patch_size, patch_size]
        hc, wc = self.hc, self.wc
        b = x.shape[0] // (hc * wc)
        c = x.shape[1]
        ps = self.patch_size
        x = x.reshape(b, hc, wc, c, ps, ps)
        x = x[..., ps // 4:ps // 4 * 3, ps // 4:ps // 4 * 3]
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, hc * ps // 2, wc * ps // 2)
        return x

    def forward(self, x):
        x = self.divide(x)
        x = self.encode(x)
        # x = self.measure(x)
        x = self.decode(x)
        x = self.merge(x)
        x = self.sigmoid(self.final_conv(x))
        return x


class InterpolateAE_T(AE):
    def __init__(self, gray=False):
        super(InterpolateAE_T, self).__init__()
        self.in_dim = 1 if gray else 3
        self.em_dim = 1 if gray else 4

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_dim, 16, 3, padding=1),  # 3*128*128 -> 16*128*128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*128*128 -> 16*64*64
            nn.Conv2d(16, 16, 3, padding=1),  # 16*64*64 -> 16*64*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*64*64 -> 16*32*32
            nn.Conv2d(16, 16, 3, padding=1),  # 16*32*32 -> 4*32*32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*32*32 -> 16*16*16
            nn.Conv2d(16, self.em_dim, 3, padding=1),  # 16*16*16 -> 4*16*16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4*16*16 -> 4*8*8
        )

        self.decoder = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),  # 4*8*8 -> 4*16*16
            nn.Conv2d(self.em_dim, 16, 3, padding=1),  # 4*16*16 -> 16*16*16
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*16*16 -> 16*32*32
            nn.Conv2d(16, 16, 3, padding=1),  # 16*32*32 -> 16*32*32
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*32*32 -> 16*64*64
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),  # 16*64*64 -> 16*128*128
            nn.Conv2d(16, self.in_dim, 3, padding=1),  # 16*128*128 -> 8*128*128
            nn.Sigmoid()
        )
