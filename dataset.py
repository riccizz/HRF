import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from utils import infiniteloop


class ImageNet32Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory of images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse the directory structure to collect image paths and labels
        for class_idx, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.endswith(".png") or img_file.endswith(".JPEG"):
                        self.image_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(class_idx)  # Assign numeric labels based on class folder order

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


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
    elif ds == 'imagenet':
        if imagenet_root is None:
            raise ValueError("imagenet_root must be provided.")
        root_dir = imagenet_root # '/data01/yichi5/imagenet32_images'
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])
        dataset = ImageNet32Dataset(
            root_dir=root_dir, transform=transform
        )
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
