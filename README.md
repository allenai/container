# container
# container

This repository contains PyTorch evaluation code, training code and pretrained models for the following projects:
* DeiT (Data-Efficient Image Transformers) 
* [CaiT](README_cait.md) (Going deeper with Image Transformers)
* [ResMLP](README_resmlp.md) (ResMLP: Feedforward networks for image classification with data-efficient training)

They obtain competitive tradeoffs in terms of speed / precision:

![DeiT](.github/deit.png)

For details see [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles and Hervé Jégou. 

If you use this code for a paper please cite:

```
@InProceedings{pmlr-v139-touvron21a,
  title =     {Training data-efficient image transformers &amp; distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}
```

# Model Zoo

We provide baseline DeiT models pretrained on ImageNet 2012.

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| DeiT-tiny | 72.2 | 91.1 | 5M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| DeiT-small | 79.9 | 95.0 | 22M| [model](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| DeiT-base | 81.8 | 95.6 | 86M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |
| DeiT-tiny distilled | 74.5 | 91.9 | 6M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth) |
| DeiT-small distilled | 81.2 | 95.4 | 22M| [model](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth) |
| DeiT-base distilled | 83.4 | 96.5 | 87M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth) |
| DeiT-base 384 | 82.9 | 96.2 | 87M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth) |
| DeiT-base distilled 384 (1000 epochs) | 85.2 | 97.2 | 88M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth) |
|CaiT-S24 distilled 384| 85.1 | 97.3 | 47M | [model](README_cait.md)|
|CaiT-M48 distilled 448| 86.5 | 97.7 | 356M | [model](README_cait.md)|

The models are also available via torch hub.
Before using it, make sure you have the pytorch-image-models package [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman) installed. Note that our work relies of the augmentations proposed in this library. In particular, the RandAugment and RandErasing augmentations that we invoke are the improved versions from the timm library, which already led the timm authors to report up to 79.35% top-1 accuracy with Imagenet training for their best model, i.e., an improvement of about +1.5% compared to prior art. 

To load DeiT-base with pretrained weights on ImageNet simply do:

```python
import torch
# check you have the right version of timm
import timm
assert timm.__version__ == "0.3.2"

# now load it with torchhub
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
```

Additionnally, we provide a [Colab notebook](https://colab.research.google.com/github/facebookresearch/deit/blob/colab/notebooks/deit_inference.ipynb) which goes over the steps needed to perform inference with DeiT.

# Usage

First, clone the repository locally:
```
git clone https://github.com/facebookresearch/deit.git
```
Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Evaluation
To evaluate a pre-trained DeiT-base on ImageNet val with a single GPU run:
```
python main.py --eval --resume https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --data-path /path/to/imagenet
```
This should give
```
* Acc@1 81.846 Acc@5 95.594 loss 0.820
```

For Deit-small, run:
```
python main.py --eval --resume https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --model deit_small_patch16_224 --data-path /path/to/imagenet
```
giving
```
* Acc@1 79.854 Acc@5 94.968 loss 0.881
```



## Training
To train Container-Light on ImageNet on a single node with 8 gpus for 300 epochs run:

Container-Light
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --batch-size 128 --data-path /path/to/imagenet --output_dir /path/to/save
```


# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Acknowledgement
Container codebase is highly motivated by DeiT
