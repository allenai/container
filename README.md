# Container : Context Aggregation Network

If you use this code for a paper please cite:

```
@article{gao2021container,
  title={Container: Context Aggregation Network},
  author={Gao, Peng and Lu, Jiasen and Li, Hongsheng and Mottaghi, Roozbeh and Kembhavi, Aniruddha},
  journal={arXiv preprint arXiv:2106.01401},
  year={2021}
}
```

# Model Zoo

We provide baseline Container-light models pretrained on ImageNet 2012.

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| Container-Light | 82.3 | 96.2 | 21M | [model](https://drive.google.com/file/d/1WMOWoxTX7AQDCbfMYh7naqIHube3K85A/view?usp=sharing) |


# Usage

First, clone the repository locally:
```
git clone https://github.com/gaopengcuhk/ContainerV1.git
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
python main.py --eval --resume checkpoint.pth --data-path /path/to/imagenet
```
This should give
```
* Acc@1 81.846 Acc@5 95.594 loss 0.820
```

For Deit-small, run:
```
python main.py --eval --resume checkpoint.pth --model container_v1_light --data-path /path/to/imagenet
```
giving
```
* Acc@1 82.3 Acc@5 96.2
```


## Training
To train DeiT-small and Deit-tiny on ImageNet on a single node with 4 gpus for 300 epochs run:

DeiT-small
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_small_patch16_224 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```

DeiT-tiny
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```


### Multinode training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train DeiT-base model on ImageNet on 2 nodes with 8 gpus each for 300 epochs:

```
python run_with_submitit.py --model deit_base_patch16_224 --data-path /path/to/imagenet
```

To train DeiT-base with hard distillation using a RegNetY-160 as teacher, on 2 nodes with 8 GPUs with 32GB each for 300 epochs (make sure that the model weights for the teacher have been downloaded before to the correct location, to avoid multiple workers writing to the same file):
```
python run_with_submitit.py --model deit_base_distilled_patch16_224 --distillation-type hard --teacher-model regnety_160 --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth --use_volta32
```

To finetune a DeiT-base on 384 resolution images for 30 epochs, starting from a DeiT-base trained on 224 resolution images, do (make sure that the weights to the original model have been downloaded before, to avoid multiple workers writing to the same file):
```
python run_with_submitit.py --model deit_base_patch16_384 --batch-size 32 --finetune https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --input-size 384 --use_volta32 --nodes 2 --lr 5e-6 --weight-decay 1e-8 --epochs 30 --min-lr 5e-6
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
