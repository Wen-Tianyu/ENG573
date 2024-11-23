import torch
import torch.nn.functional as F


def batched_ops(op, batched_pred, batched_target):
    return torch.tensor([
        op(pred, target) for pred, target in zip(batched_pred, batched_target)]
    )


def MSE(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    return (pred - target).norm() ** 2 / pred.size(0)


def NMSE(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    return ((pred - target).norm() / target.norm()) ** 2


def PSNR(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    return 10 * torch.log10((target ** 2).max() / MSE(pred, target))


def SSIM(pred, target, return_lcs=False, L=255, K1=0.01, K2=0.01):
    pred = pred.view(-1)
    target = target.view(-1)

    mean_pred = pred.mean()
    mean_target = target.mean()

    std_pred = pred.std()
    std_target = target.std()
    cov_pred = ((pred - mean_pred) * (target - mean_target)).sum() / (pred.size(0) - 1)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2
    luminance = (2 * mean_pred * mean_target + C1) / (mean_target ** 2 + mean_pred ** 2 + C1)
    contrast = (2 * std_pred * std_target + C2) / (std_target ** 2 + std_pred ** 2 + C2)
    structure_comparison = (cov_pred + C3) / (std_pred * std_target + C3)

    ssim = luminance * contrast * structure_comparison
    if return_lcs:
        return {'l': luminance,
                'c': contrast,
                's': structure_comparison}
    else:
        return ssim


def gaussian_kernel(size=3, sigma=1.0):
    x_coord = torch.arange(size)
    x_grid = x_coord.repeat(size).view(size, size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (size - 1) / 2.
    variance = sigma ** 2.

    kernel = (1. / (2. * torch.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
    )

    kernel = kernel / torch.sum(kernel)
    return kernel


def gaussian_filter_image(image, size=3, sigma=1.0):
    kernel = gaussian_kernel(size=size, sigma=sigma)
    kernel = kernel[None, None, ...]
    image = image[None, None, ...]
    filtered_image = F.conv2d(image, kernel, padding=1).squeeze()

    return filtered_image


def MS_SSIM(pred, target, M=3, L=255, K1=0.01, K2=0.01,
            gaussian_size=3, gaussian_sigma=1.0):
    lcs = SSIM(pred, target, return_lcs=True, L=L, K1=K1, K2=K2)
    L = lcs['l']
    C = lcs['c']
    S = lcs['s']

    for j in range(M - 1):
        pred = gaussian_filter_image(pred, size=gaussian_size, sigma=gaussian_sigma)
        pred = F.interpolate(pred[None, None, ...], size=(pred.size(-2) // 2, pred.size(-1) // 2),
                             mode='bilinear').squeeze()
        target = gaussian_filter_image(target, size=gaussian_size, sigma=gaussian_sigma)
        target = F.interpolate(target[None, None, ...], size=(target.size(-2) // 2, target.size(-1) // 2),
                               mode='bilinear').squeeze()

        lcs = SSIM(pred, target, return_lcs=True, L=L, K1=K1, K2=K2)
        L = lcs['l']
        C *= lcs['c']
        S *= lcs['s']

    return (L * C * S) ** (1 / M)
