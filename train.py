import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from functools import partial

from utils.dataloader import datainfo, dataload
from model.vit import ViT
from utils.loss import LabelSmoothingCrossEntropy
from utils.scheduler import build_scheduler  
from utils.optimizer import get_adam_optimizer
from utils.utils import clip_gradients
from utils.utils import save_checkpoint, load_checkpoint


def get_args_parser():
    parser = argparse.ArgumentParser('ViT for CIFAR-10', add_help=False)
    parser.add_argument('--dir', type=str, default='./data',
                    help='Data directory')
    parser.add_argument('--num_classes', type=int, default=10, choices=[10, 100, 1000],
                    help='Dataset name')

    # Model parameters
    parser.add_argument('--patch_size', default=2, type=int, help="""Size in pixels of input square patches - default 4 (for 4x4 patches) """)
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of the SSL MLP head output. For complex and large datasets large values (like 65k) work well.""")

    parser.add_argument('--norm_last_layer', default=False, type=bool,
        help="""Whether or not to weight normalize the last layer of the MLP head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--image_size', default=32, type=int, help=""" Size of input image. """)
    parser.add_argument('--in_channels',default=3, type=int, help=""" input image channels. """)
    parser.add_argument('--embed_dim',default=192, type=int, help=""" dimensions of vit """)
    parser.add_argument('--num_layers',default=9, type=int, help=""" No. of layers of ViT """)
    parser.add_argument('--num_heads',default=12, type=int, help=""" No. of heads in attention layer
                                                                                 in ViT """)
    parser.add_argument('--vit_mlp_ratio',default=2, type=int, help=""" MLP hidden dim """)
    parser.add_argument('--qkv_bias',default=True, type=bool, help=""" Bias in Q K and V values """)
    parser.add_argument('--drop_rate',default=0., type=float, help=""" dropout """)

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=1e-1, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--batch_size', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. Recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing for optimizer')
    parser.add_argument('--gamma', type=float, default=1.0,
                    help='Gamma value for Cosine LR schedule')
    parser.add_argument('--channels', type=int, default=256,
                    help='Embedding dimension')
    parser.add_argument('--head_channels', type=int, default=32,
                    help='Head embedding dimension')
    parser.add_argument('--num_blocks', type=int, default=8,
                    help='Number of transformer blocks')

    # Misc
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100'], help='Please specify path to the training data.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--mlp_head_in", default=192, type=int, help="input dimension going inside MLP projection head")

    return parser



class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, lr_scheduler, loss, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.device = device
        self.args = args

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def train(self):
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        best_accuracy = 0.0

        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss, total_correct = 0.0, 0

            train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs} [Training]", total=len(train_loader))
            for images, labels in train_progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                if self.args.clip_grad > 0:
                    self.clip_gradients()

                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                train_progress_bar.set_postfix({"Train Loss": total_loss / (train_progress_bar.n + 1)})

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = total_correct / len(train_loader.dataset)
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            self.model.eval()
            total_loss, total_correct = 0.0, 0
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs} [Validation]", total=len(val_loader))
            with torch.no_grad():
                for images, labels in val_progress_bar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == labels).sum().item()
                    val_progress_bar.set_postfix({"Val Loss": total_loss / (val_progress_bar.n + 1)})

            avg_val_loss = total_loss / len(val_loader)
            val_accuracy = total_correct / len(val_loader.dataset)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            self.logger.info(f"Epoch {epoch + 1}/{self.args.epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), "best_model.pth")
                self.logger.info(f"New best accuracy: {best_accuracy:.4f}, Model saved as 'best_model.pth'")

            self.lr_scheduler.step()

            # Save checkpoint at the end of each epoch
            self.save_checkpoint(epoch)

        return train_losses, val_losses, train_accuracies, val_accuracies


def main():
    args, unknown = get_args_parser().parse_known_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_info = datainfo(args)
    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]

    train_dataset, val_dataset = dataload(args, normalize, data_info)   

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                                num_workers=args.num_workers, pin_memory=True)

    model = ViT(img_size=[args.image_size],
            patch_size=args.patch_size,
            in_chans=args.in_channels,
            num_classes=0,
            embed_dim=192,
            depth=9,
            num_heads=12,
            mlp_ratio=2,
            qkv_bias=args.qkv_bias,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)

    loss = LabelSmoothingCrossEntropy()
    optimizer = get_adam_optimizer(model.parameters(), lr=args.lr, wd=args.weight_decay)
    lr_scheduler = build_scheduler(args, optimizer)
    
    Trainer(model, train_loader, val_loader, optimizer, lr_scheduler, loss, device, args).train()

if __name__ == "__main__":
    main()
