# imagenet train from scratch command
# python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data1/home/hseo0618/src/imagenet/dataset/imagenet

# imagenet train command for resume
# python main.py -a resnet50 --resume /data1/home/hseo0618/src/imagenet/code/base/checkpoint.pth.tar --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data1/home/hseo0618/src/imagenet/dataset/imagenet


# imagenet train from scratch command
# python main_sdkd.py -a resnet50 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data1/home/hseo0618/src/imagenet/dataset/imagenet
# python main_sdkd.py -a resnet50 --dist-url 'tcp://127.0.0.1:9998' --resume /data1/home/hseo0618/src/imagenet/code/base/kd_checkpoint.pth.tar --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data1/home/hseo0618/src/imagenet/dataset/imagenet
