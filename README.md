```bash
# CIFAR-10
CUDA_VISIBLE_DEVICES=0 python train.py --cutout --auxiliary --data /raid/ --arch TENAS_cifar10 --batch_size 96 --epoch 800

# ImageNet dist
python train_imagenet_dist.py --batch_size 768 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --data /raid/imagenet --arch TENAS_imagenet --epochs 350
```
