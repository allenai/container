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
git clone https://github.com/allenai/container.git
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
To evaluate a pre-trained Container-Light on ImageNet val with a single GPU run:


For Container-Light, run:
```
python main.py --eval --resume checkpoint.pth --model container_v1_light --data-path /path/to/imagenet
```
giving
```
* Acc@1 82.26
```



## Training
To train Container-Light on ImageNet on a single node with 8 gpus for 300 epochs run:

Container-Light
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model container_v1_light --batch-size 128 --data-path /path/to/imagenet --output_dir /path/to/save
```
## Downstream task on SMCA-DETR, Retinanet and Mask RCNN
Code will be released seperately. 

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Acknowledgement
Container codebase is highly motivated by DeiT
