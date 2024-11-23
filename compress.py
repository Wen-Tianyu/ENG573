import time

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import quant
from data import MyDrillingDataset
from model import InterpolateAE, InterpolateAE_S, InterpolateAE_T
from utils import load_checkpoint

if __name__ == '__main__':
    suffix = "a"
    epoch = 100
    batch_size = 16
    transform = transforms.Compose([transforms.ToTensor()])

    compress_dataset = MyDrillingDataset("./dataset/mydrilling_aug/test", transform=transform)
    compress_loader = DataLoader(dataset=compress_dataset, batch_size=batch_size, shuffle=False)

    model = InterpolateAE()
    if suffix.startswith('s'):
        model = InterpolateAE_S()
    if suffix.startswith('t'):
        model = InterpolateAE_T()
    model.load_state_dict(load_checkpoint(epoch, suffix=suffix)[0])
    model.eval()

    times = 0
    with torch.no_grad():
        L = []
        for data in tqdm(compress_loader):
            imgs = data
            start_time = time.time()
            outputs = model.compress(imgs) - 127
            end_time = time.time()
            times += end_time - start_time
            L.append(len(quant.get_bit_strings(outputs)) / imgs.shape[0])
        plt.plot(L)
        plt.xlabel('steps')
        plt.ylabel('#bits')
        plt.savefig('./outputs_before/compressed_l_{}.png'.format(suffix))
        plt.close()
    print('time to compress one image:', times / len(compress_dataset))
    print('average #bits per image (KB):', np.mean(L) / 1024)
