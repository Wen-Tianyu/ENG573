import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import MNIST, CIFAR10

import metrics
from data import MyDrillingDataset
from model import *
from utils import save_checkpoint, load_checkpoint

if __name__ == '__main__':
    dataset = 'd'  # d for drilling, m for MNIST, c for CIFAR10
    suffix = 'a_ms'  # (a)standard, (s)small, (t)tiny, _ft(MNIST finetune), _ftc(CIFAR10 finetune), _ms(multi-stage)
    gray = False
    if gray:
        suffix += '_gray'

    batch_size = 32
    lr = 5e-4
    wd = 1e-3
    last_epoch = 0
    num_epochs = 100
    cp_epochs = 10 if dataset == 'd' else 1

    if dataset == 'd':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = MyDrillingDataset("./dataset/mydrilling_aug/train", transform=transform, gray=gray)
    elif dataset == 'm':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((128, 128))])
        train_dataset = MNIST("./dataset", train=True, download=True, transform=transform)
    elif dataset == 'c':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((128, 128))])
        train_dataset = CIFAR10("./dataset/CIFAR10", train=True, download=True, transform=transform)
    else:
        raise NotImplementedError
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if last_epoch > 0:
        model_sd, optimizer_sd = load_checkpoint(last_epoch, suffix=suffix)
        model.load_state_dict(model_sd)
        optimizer.load_state_dict(optimizer_sd)

    MSEs, NMSEs, PSNRs, SSIMs = [], [], [], []
    for epoch in range(max(0, last_epoch) + 1, num_epochs + 1):
        for data in tqdm(train_loader):
            if dataset == 'd':
                imgs = data
            elif dataset == 'm':
                imgs = data[0].repeat(1, 3, 1, 1)
            elif dataset == 'c':
                imgs = data[0]
            else:
                raise NotImplementedError
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            MSEs.append(metrics.batched_ops(metrics.MSE, outputs.squeeze(), imgs.squeeze()).mean().item())
            NMSEs.append(metrics.batched_ops(metrics.NMSE, outputs.squeeze(), imgs.squeeze()).mean().item())
            PSNRs.append(metrics.batched_ops(metrics.PSNR, outputs.squeeze(), imgs.squeeze()).mean().item())
            SSIMs.append(metrics.batched_ops(metrics.SSIM, outputs.squeeze(), imgs.squeeze()).mean().item())

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
        if epoch % cp_epochs == 0:
            save_checkpoint(epoch, model, optimizer, suffix=suffix)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(MSEs, label='MSE')
    axes[0, 0].set_title("MSE")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_xlabel("Step")

    axes[0, 1].plot(NMSEs, label='NMSE')
    axes[0, 1].set_title("NMSE")
    axes[0, 1].set_ylabel("NMSE")
    axes[0, 1].set_xlabel("Step")

    axes[1, 0].plot(PSNRs, label='PSNR')
    axes[1, 0].set_title("PSNR")
    axes[1, 0].set_ylabel("PSNR")
    axes[1, 0].set_xlabel("Step")

    axes[1, 1].plot(SSIMs, label='SSIM')
    axes[1, 1].set_title("SSIM")
    axes[1, 1].set_ylabel("SSIM")
    axes[1, 1].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig('./outputs_before/steps_metrics_{}.png'.format(suffix))
    plt.close()
