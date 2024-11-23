import torch


class Quanter:
    def __init__(self):
        self.quanter = lambda x: x
        self.dequanter = lambda x: x

    def quant(self, x):
        return self.quanter(x)

    def dequant(self, x):
        return self.dequanter(x)


class Quanter_N(Quanter):
    def __init__(self, N=8):
        super().__init__()
        self.N = N
        self.quanter = lambda n: (n * (2 ** self.N) - 0.5).to(torch.int)
        self.dequanter = lambda n: ((n - 0.5) / (2 ** self.N)).to(torch.float32)


class Quanter_S(Quanter):
    def __init__(self):
        super().__init__()
        self.quanter = torch.sign


def get_bit_string(n: int):
    return bin(n)[2:]


def get_bit_strings(x):
    assert x.dtype == torch.int
    x = x.flatten()
    return '.'.join([get_bit_string(n.item()) for n in x])
