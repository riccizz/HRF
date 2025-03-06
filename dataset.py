import torch
from torchvision import datasets, transforms

from utils import infiniteloop


def get_datalooper(ds, batch_size, num_workers, train=True, imagenet_root=None):
    if ds == 'cifar10':
        dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        data_shape = (3, 32, 32)
    elif ds == 'mnist':
        dataset = datasets.MNIST(
            "./data",
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5,), (0.5,))
                ]
            ),
        )
        data_shape = (1, 28, 28)
    elif ds == 'imagenet32':
        if imagenet_root is None:
            raise ValueError("imagenet_root must be provided.")
        root_dir = imagenet_root 
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])
        dataset = datasets.ImageFolder(root_dir, transform)
        data_shape = (3, 32, 32)
    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=True,
    )
    datalooper = infiniteloop(dataloader)

    return datalooper, data_shape
