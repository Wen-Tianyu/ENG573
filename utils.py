import torch


def save_checkpoint(epoch, model, optimizer, path="checkpoint/", suffix=""):
    filename = f"{path}model_{suffix}_e{epoch}.pt"
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               filename)


def load_checkpoint(epoch, path="checkpoint/", suffix=""):
    filename = f"{path}model_{suffix}_e{epoch}.pt"
    load_dict = torch.load(filename, map_location=torch.device('cpu'))
    model = load_dict['model']
    optimizer = load_dict['optimizer']

    return model, optimizer
