import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.RandomErasing(p=0.1)
])

train_dset = datasets.CIFAR10(root=args.DATA_DIR, train=True, download=True, transform=train_transform)
test_dset = datasets.CIFAR10(root=args.DATA_DIR, train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.BATCH_SIZE, shuffle=True,
                                           num_workers=args.NUM_WORKERS, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.BATCH_SIZE, shuffle=False,
                                          num_workers=args.NUM_WORKERS, pin_memory=True)