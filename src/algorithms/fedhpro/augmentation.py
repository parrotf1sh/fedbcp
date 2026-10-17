import numpy as np
import torch
import torch.nn.functional as F


class AugmentParams:
    def __init__(self, settings):
        for name, value in settings.items():
            setattr(self, name, value)
        self.siamese = False
        self.latest_seed = -1


def diff_augment(x, strategy="", seed=-1, param=None):
    if seed == -1:
        param.siamese = False
    else:
        param.siamese = True

    param.latest_seed = seed

    if strategy == "None" or strategy == "none":
        return x

    if strategy:
        if param.aug_mode == "M":
            for name in strategy.split("_"):
                for augment in AUGMENT_FNS[name]:
                    x = augment(x, param)
        elif param.aug_mode == "S":
            strategies = strategy.split("_")
            set_augment_seed(param)
            name = strategies[torch.randint(0, len(strategies), size=(1,)).item()]
            for augment in AUGMENT_FNS[name]:
                x = augment(x, param)
        else:
            raise ValueError("Unknown augmentation mode: {}".format(param.aug_mode))
        x = x.contiguous()
    return x


def rand_scale(x, param):
    ratio = param.ratio_scale
    set_augment_seed(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    set_augment_seed(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = [
        [[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])
    ]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.siamese:
        aux = theta.clone()
        theta[:] = aux[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param):
    ratio = param.ratio_rotate
    set_augment_seed(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [
        [[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
         [torch.sin(theta[i]), torch.cos(theta[i]), 0]]
        for i in range(x.shape[0])
    ]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.siamese:
        aux = theta.clone()
        theta[:] = aux[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_augment_seed(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.siamese:
        aux = randf.clone()
        randf[:] = aux[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_augment_seed(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.siamese:  # Siamese augmentation:
        aux = randb.clone()
        randb[:] = aux[0]
    x = x + (randb - 0.5) * ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_augment_seed(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.siamese:  # Siamese augmentation:
        aux = rands.clone()
        rands[:] = aux[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_augment_seed(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.siamese:  # Siamese augmentation:
        aux = randc.clone()
        randc[:] = aux[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # padded on its surrounding and then cropped
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_augment_seed(param)
    translation_x = torch.randint(
        -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
    )
    set_augment_seed(param)
    translation_y = torch.randint(
        -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    )
    if param.siamese:  # Siamese augmentation:
        aux_x = translation_x.clone()
        aux_y = translation_y.clone()
        translation_x[:] = aux_x[0]
        translation_y[:] = aux_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_augment_seed(param)
    offset_x = torch.randint(
        0, x.size(2) + (1 - cutout_size[0] % 2),
        size=[x.size(0), 1, 1], device=x.device,
    )
    set_augment_seed(param)
    offset_y = torch.randint(
        0, x.size(3) + (1 - cutout_size[1] % 2),
        size=[x.size(0), 1, 1], device=x.device,
    )
    if param.siamese:  # Siamese augmentation:
        aux_x = offset_x.clone()
        aux_y = offset_y.clone()
        offset_x[:] = aux_x[0]
        offset_y[:] = aux_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "crop": [rand_crop],
    "cutout": [rand_cutout],
    "flip": [rand_flip],
    "scale": [rand_scale],
    "rotate": [rand_rotate],
}


def set_augment_seed(param):
    if param.latest_seed == -1:
        return
    else:
        torch.random.manual_seed(param.latest_seed)
        param.latest_seed += 1
