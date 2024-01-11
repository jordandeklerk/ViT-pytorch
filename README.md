# Training Vision Transformer on Small-scale Datasets (CIFAR-10)

#
<hr>

## Contents

1. [Highlights](#Highlights)
2. [Requirements](#Requirements)
3. [Supervised Training](#Training)
4. [Results](#Results)


<hr>

## Highlights
This project is a implementation from scratch of a slightly modified version of the vanilla vision transformer introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). We implement this model on the small scale benchmark dataset `CIFAR-10`. 

Vision Transformers often suffer when trained from scratch on small datasets such as `CIFAR-10`. This is primarily due to the lack of locality, inductive biases and hierarchical structure of the representations which is commonly observed in the Convolutional Neural Networks. As a result, ViTs require large-scale pre-training to learn such properties from the data for better transfer learning to downstream tasks. This project shows that with modifications, supervised training of vision transformer models on small scale datasets like `CIFAR-10` can lead to very high accuracy with low computational constraints. 

<img src="./Images/vit.gif" width="500px"></img>

The vanilla vision transformer model uses the standard multi-head self-attention mechanism introduced in the seminal paper by [Vaswani et al.](https://arxiv.org/abs/1706.03762). We introduce a slightly modified version of self-attention using convolutional projections for keys, values, and queries as opposed to the standard linear projeciton. This allows us to capture more of the spatial context of the images. Our self-attention module is given by the following:

```python
class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, head_channels, shape):
        super().__init__()
        self.heads = out_channels // head_channels
        self.head_channels = head_channels
        self.scale = head_channels**-0.5

        self.to_keys = nn.Conv2d(in_channels, out_channels, 1)
        self.to_queries = nn.Conv2d(in_channels, out_channels, 1)
        self.to_values = nn.Conv2d(in_channels, out_channels, 1)
        self.unifyheads = nn.Conv2d(out_channels, out_channels, 1)

        height, width = shape
        self.pos_enc = nn.Parameter(torch.Tensor(self.heads, (2 * height - 1) * (2 * width - 1)))
        self.register_buffer("relative_indices", self.get_indices(height, width))

    def forward(self, x):
        b, _, h, w = x.shape

        keys = self.to_keys(x).view(b, self.heads, self.head_channels, -1)
        values = self.to_values(x).view(b, self.heads, self.head_channels, -1)
        queries = self.to_queries(x).view(b, self.heads, self.head_channels, -1)

        att = keys.transpose(-2, -1) @ queries

        indices = self.relative_indices.expand(self.heads, -1)
        rel_pos_enc = self.pos_enc.gather(-1, indices)
        rel_pos_enc = rel_pos_enc.unflatten(-1, (h * w, h * w))

        att = att * self.scale + rel_pos_enc
        att = F.softmax(att, dim=-2)

        out = values @ att
        out = out.view(b, -1, h, w)
        out = self.unifyheads(out)
        return out

    @staticmethod
    def get_indices(h, w):
        y = torch.arange(h, dtype=torch.long)
        x = torch.arange(w, dtype=torch.long)

        y1, x1, y2, x2 = torch.meshgrid(y, x, y, x, indexing='ij')
        indices = (y1 - y2 + h - 1) * (2 * w - 1) + x1 - x2 + w - 1
        indices = indices.flatten()

        return indices
```


<hr>

## Requirements
```shell
pip install -r requirements.txt
```

<hr>

## Supervised Training
This repository is intended to be run in a notebook rather than from the command line. The organization of the files is intended to break apart `main.py` to highlight the different components. Future work will be completed to turn this repository into a working directory.

The main hyperparamerters used for training and inference are listed below. The full set of hyperparamters can be found in `parser.py`.
```shell
python main.py --dataset CIFAR-10 \
               --datapath "/path/to/data/folder" \
               --batch_size 128 \
               --epochs 200 \
               --learning rate 1e-3 \
               --weight decay 1e-1 \
               --min lr 1e-5 \
               --warm-up epochs 10 \
```

<hr>

## Results
We test our approach on the `CIFAR-10` dataset with the intention to extend our model to 4 other small low resolution datasets: `Tiny-Imagenet`, `CIFAR100`, `CINIC10` and `SVHN`. All training took place on a single V100 GPU with total training time taking approximately 21617s. We have included the notebook in this repository that can be downloaded and run in any environment with access to a GPU.
  * CIFAR10
    * ```vit_cifar10_patch2_input32``` - 96.80 @ 32
