import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data import MyDrillingDataset
from model import *
from utils import *
import metrics

if __name__ == '__main__':
    suffix = 'a_hp'
    epoch = 10
    batch_size = 6
    gray = False
    if gray:
        suffix += '_gray'

    transform = transforms.Compose([transforms.ToTensor()])
    inf_dataset = MyDrillingDataset("./dataset/mydrilling_aug/test", transform=transform, gray=gray)
    test_loader = DataLoader(dataset=inf_dataset, batch_size=batch_size, shuffle=False)

    model_cls = InterpolateAE
    if suffix.startswith('s'):
        model_cls = InterpolateAE_S
    if suffix.startswith('t'):
        model_cls = InterpolateAE_T
    if suffix == 's_ms':
        model_cls = InterpolateAE_S_MS
    if suffix == 'a_ms':
        model_cls = InterpolateAE_MS
    model = model_cls(gray=gray)
    model.load_state_dict(load_checkpoint(epoch, suffix=suffix)[0])
    model.eval()

    MSEs, NMSEs, PSNRs, SSIMs = [], [], [], []
    with torch.no_grad():
        i = 0
        for data in tqdm(test_loader):
            imgs = data
            outputs = model(imgs)

            s = 1
            if i % s == 0:
                target_0 = imgs[0].permute(1, 2, 0).squeeze()
                pred_0 = outputs[0].permute(1, 2, 0).squeeze()
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                axs[0].imshow(target_0)
                axs[1].imshow(pred_0)
                axs[0].axis('off')
                axs[1].axis('off')
                plt.tight_layout()
                plt.savefig('./outputs/{}_{}_e{}.png'.format(i // s, suffix, epoch))
                plt.close()

                # target_hp = edge_filter(imgs)
                # pred_hp = edge_filter(outputs)
                # target_hp = target_hp[0].permute(1, 2, 0).squeeze()
                # pred_hp = pred_hp[0].permute(1, 2, 0).squeeze()
                # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                # axs[0].imshow(target_hp)
                # axs[1].imshow(pred_hp)
                # axs[0].axis('off')
                # axs[1].axis('off')
                # plt.tight_layout()
                # plt.savefig('./outputs/{}_hp.png'.format(i // s))
            i += 1

            MSEs.append(metrics.batched_ops(metrics.MSE, outputs.squeeze(), imgs.squeeze()).mean().item())
            NMSEs.append(metrics.batched_ops(metrics.NMSE, outputs.squeeze(), imgs.squeeze()).mean().item())
            PSNRs.append(metrics.batched_ops(metrics.PSNR, outputs.squeeze(), imgs.squeeze()).mean().item())
            SSIMs.append(metrics.batched_ops(metrics.SSIM, outputs.squeeze(), imgs.squeeze()).mean().item())

    print(f'MSE: {np.mean(MSEs)}')
    print(f'NMSE: {np.mean(NMSEs)}')
    print(f'PSNR: {np.mean(PSNRs)}')
    print(f'SSIM: {np.mean(SSIMs)}')
