import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def collate(batch):
    images, labels = zip(*batch)
    images = [np.array(img) for img in images]
    images = np.array(images)
    labels = torch.tensor(labels)
    return torch.tensor(images).permute(0, 3, 1, 2), labels

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.BATCH_SIZE, shuffle=True, collate_fn=collate)



def imshow(img):
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.axis('off')

dataiter = iter(trainloader)
images, labels = next(dataiter)

class_images = {class_label: [] for class_label in range(10)}
for img, label in zip(images, labels):
    if len(class_images[label.item()]) < 10:
        class_images[label.item()].append(img.numpy())

plt.figure(figsize=(8,8))
sns.set_style('white')

for class_idx, class_label in enumerate(classes):
    plt.subplot(11, 11, class_idx * 11 + 1)
    plt.text(0.5, 0.5, class_label, ha='center', va='center', fontsize=12)
    plt.axis('off')

for class_idx, (class_label, img_list) in enumerate(class_images.items()):
    for img_num, img in enumerate(img_list):
        plt.subplot(11, 11, class_idx * 11 + img_num + 2)
        imshow(img)
plt.show()




NUM_IMAGES = 4
CIFAR_images = torch.stack([test_dset[idx][0] for idx in range(NUM_IMAGES)], dim=0)
img_grid = torchvision.utils.make_grid(CIFAR_images, nrow=4, normalize=True, pad_value=0.9)
img_grid = img_grid.permute(1, 2, 0).numpy()

sns.set(style="whitegrid", context="talk", palette="rainbow")

plt.figure(figsize=(8, 8))
plt.title("Image examples of the CIFAR10 dataset")
plt.imshow(img_grid)
plt.axis("off")
plt.show()
plt.close()




def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the image of shape [B, C, H, W]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

img_patches = img_to_patch(CIFAR_images, patch_size=2, flatten_channels=False)

sns.set(style="whitegrid", context="talk", palette="rainbow")

fig, ax = plt.subplots(CIFAR_images.shape[0], 1, figsize=(12, 8))
fig.suptitle("Images as input sequences of patches")

for i in range(CIFAR_images.shape[0]):
    img_grid = torchvision.utils.make_grid(img_patches[i], nrow=64, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0).numpy()

    ax[i].imshow(img_grid)
    ax[i].axis("off")

plt.show()
plt.close()



lrs = []
for epoch in range(200):
    lr_scheduler.step(epoch)
    lrs.append(lr_scheduler.get_lr())

plt.figure(figsize=(8,6))
plt.plot(lrs)
plt.title('Cosine Annealing Learning Rate Scheduler')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()



def plot_cutmix_images(model, data_loader, device):

    images, labels = next(iter(data_loader))
    mixed_images, _ = model.prepare_batch((images, labels), device=DEVICE, non_blocking=True)
    np_images = mixed_images.cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i, ax in enumerate(axes):
        # Transpose the image from (C, H, W) to (H, W, C) and normalize
        img = np.transpose(np_images[i], (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())

        ax.imshow(img)
        ax.axis('off')
    plt.show()

cutmix_model = CutMix(loss = loss, α = args.ALPHA)
cifar10_loader = torch.utils.data.DataLoader(train_dset, batch_size=5, shuffle=True,
                                           num_workers=args.NUM_WORKERS, pin_memory=True)

plot_cutmix_images(cutmix_model, cifar10_loader, device=DEVICE)