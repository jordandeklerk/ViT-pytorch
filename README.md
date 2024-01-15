# Vision Transformer on CIFAR-10

<hr>

## Contents

1. [Highlights](#Highlights)
2. [Requirements](#Requirements)
3. [Usage](#Usage)
4. [Results](#Results)


<hr>

## Highlights
This project is a implementation from scratch of a slightly modified version of the vanilla vision transformer introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). We implement this model on the small scale benchmark dataset `CIFAR-10`. 

Vision Transformers often suffer when trained from scratch on small datasets such as `CIFAR-10`. This is primarily due to the lack of locality, inductive biases and hierarchical structure of the representations which is commonly observed in the Convolutional Neural Networks. As a result, ViTs require large-scale pre-training to learn such properties from the data for better transfer learning to downstream tasks. 

This project shows that with modifications, supervised training of vision transformer models on small scale datasets like `CIFAR-10` can lead to very high accuracy with low computational constraints. 

<img src="./Images/vit.gif" width="750"></img>

The vanilla vision transformer model uses the standard multi-head self-attention mechanism introduced in the seminal paper by [Vaswani et al.](https://arxiv.org/abs/1706.03762).

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        all_head_dim = head_dim * self.num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
```


<hr>

## Requirements
```shell
pip install -r requirements.txt
```

<hr>

## Usage
To replicate the reported results, clone this repo
```shell
cd your_directory git clone git@github.com:jordandeklerk/ViT-pytorch.git
```
and run the main training script
```shell
python train.py 
```

<hr>

## Results
We test our approach on the `CIFAR-10` dataset with the intention to extend our model to 4 other small low resolution datasets: `Tiny-Imagenet`, `CIFAR100`, `CINIC10` and `SVHN`. All training took place on a single V100 GPU with total training time taking approximately 21617s.
  * CIFAR10
    * ```vit_cifar10_patch2_input32``` - 96.80 @ 32

Flop analysis:
```
total flops: 915674304
total activations: 10735212
number of parameter: 2725632
| module            | #parameters or shape   | #flops   |
|:------------------|:-----------------------|:---------|
| model             | 2.726M                 | 0.916G   |
|  cls_token        |  (1, 1, 192)           |          |
|  pos_embed        |  (1, 257, 192)         |          |
|  patch_embed.proj |  2.496K                |  0.59M   |
|  blocks           |  2.673M                |  0.915G  |
|  norm             |  0.384K                |  0.247M  |
```

<hr>

## Citations
```bibtex
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
