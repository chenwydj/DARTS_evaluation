# Evaluation Tool for architectures from DARTS search space

This a slimmed repository for evaluating architectures searched from DARTS search space (originally from NASNET search space) on CIFAR-10/100 and ImageNet, i.e., training an architecutre from scratch.

This repository is also used to evaluate the searched architectures by [TENAS](https://github.com/VITA-Group/TENAS) and [TEGNAS](https://github.com/VITA-Group/TEGNAS). Genotypes of our searched architectures are listed in [`genotypes.py`](https://github.com/chenwydj/DARTS_evaluation/blob/main/genotypes.py#L77-L82)

## Usage:

Please put your searched genotype inside `genotypes.py`.

### CIFAR-10
```bash
# CIFAR-10 training
CUDA_VISIBLE_DEVICES=0 python train.py --cutout --auxiliary --data /raid/ --arch TENAS_cifar10 --batch_size 96 --epoch 800
```

### ImageNet
The settings `batch_size = 768` and `learning_rate = 0.5` by default are designed for 8-gpu training.
```bash
# ImageNet distributed training
python train_imagenet_dist.py --batch_size 768 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --data /raid/imagenet --arch TENAS_imagenet\
--save_root ./
```
