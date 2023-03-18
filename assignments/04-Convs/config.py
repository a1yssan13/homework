from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms


class CONFIG:
    batch_size = 64 * 8
    num_epochs = 2

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=0.008)

    # optimizer_factory: Callable[
    #     [nn.Module], torch.optim.Optimizer
    # ] = lambda model: torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer_factory: Callable[
    #     [nn.Module], torch.optim.Optimizer
    # ] = lambda model: torch.optim.AdamW(model.parameters(), lr=3e-4)

    transforms = Compose(
        [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # transforms = Compose(
    #     [
    #         # transforms.RandomHorizontalFlip(),
    #         # transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             [0.49139968, 0.48215841, 0.44653091],
    #             [0.24703223, 0.24348513, 0.26158784],
    #         ),
    #     ]
    # )
