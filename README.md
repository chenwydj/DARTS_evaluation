```bash
# ImageNet dist
python train_imagenet_dist.py --batch_size 768 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --data /raid/imagenet --arch TENAS_imagenet_108 --epochs 350
```
