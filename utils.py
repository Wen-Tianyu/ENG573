import torch
import torch.nn.functional as F


def save_checkpoint(epoch, model, optimizer, scheduler, path="checkpoint/", suffix=""):
    filename = f"{path}model_{suffix}_e{epoch}.pt"
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
               filename)


def load_checkpoint(epoch, path="checkpoint/", suffix=""):
    filename = f"{path}model_{suffix}_e{epoch}.pt"
    load_dict = torch.load(filename, map_location=torch.device('cpu'))
    model = load_dict['model']
    optimizer = load_dict['optimizer']
    scheduler = load_dict['scheduler']

    return model, optimizer, scheduler


def high_pass_filter(image):
    fft = torch.fft.fft2(image, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

    _, _, h, w = image.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    center = (h // 2, w // 2)
    radius = 0.1 * min(h, w)
    mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2) > radius ** 2
    mask = mask.to(image.device).float()
    filtered_fft = fft_shift * mask[None, None, :, :]

    filtered_image = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
    high_freq_image = torch.fft.ifft2(filtered_image, dim=(-2, -1)).real
    return high_freq_image


def edge_filter(image):
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)

    image_gray = image.mean(dim=1, keepdim=True)
    grad_x = F.conv2d(image_gray, sobel_x, padding=1)
    grad_y = F.conv2d(image_gray, sobel_y, padding=1)
    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return edges


def filter_loss(filter_fn, output, target):
    output_filtered = filter_fn(output)
    target_filtered = filter_fn(target)
    loss = F.mse_loss(output_filtered, target_filtered)

    return loss
