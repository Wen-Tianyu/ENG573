import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm

from data import MyDrillingDataset
from model import *
from utils import *

if __name__ == '__main__':
    dataset = 'd'  # d for drilling, m for MNIST, c for CIFAR10
    suffix = 'a_hp'  # (a)standard, (s)small, (t)tiny, _ft(MNIST finetune), _ftc(CIFAR10 finetune), _ms(multi-stage)
    gray = False
    if gray:
        suffix += '_gray'

    batch_size = 32
    lr = 1e-3
    wd = 1e-3
    last_epoch = 0
    num_epochs = 100
    cp_epochs = 10 if dataset == 'd' else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = model_cls(gray=gray).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    warmup_epochs = 20
    initial_lr = lr
    target_lr = 5e-3

    if last_epoch > 0:
        model_sd, optimizer_sd, scheduler_sd = load_checkpoint(last_epoch, suffix=suffix)
        model.load_state_dict(model_sd)
        optimizer.load_state_dict(optimizer_sd)
        scheduler.load_state_dict(scheduler_sd)

    losses = []
    for epoch in range(max(0, last_epoch) + 1, num_epochs + 1):
        if epoch < warmup_epochs:
            warmup_lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = torch.tensor(warmup_lr)
        else:
            scheduler.step()

        for data in tqdm(train_loader):
            if dataset == 'd':
                imgs = data
            elif dataset == 'm':
                imgs = data[0].repeat(1, 3, 1, 1)
            elif dataset == 'c':
                imgs = data[0]
            else:
                raise NotImplementedError
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)

            # high-pass loss branch
            filter_weight = 0.05
            loss += filter_weight * filter_loss(edge_filter, outputs, imgs)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
        if epoch % cp_epochs == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, suffix=suffix)
