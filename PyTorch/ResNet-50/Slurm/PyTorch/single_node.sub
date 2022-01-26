#!/bin/bash
#SBATCH --job-name resnet50
#SBATCH --nodes 1

srun --container-image $1 \
 --container-mounts $2:/imagenet,$3:/pytorch \
 --no-container-entrypoint \
 /bin/bash -c \
 "python /pytorch/imagenet/main.py \
 -a resnet50 \
 --dist-url 'tcp://127.0.0.1:$4' \
 --dist-backend 'nccl' \
 --multiprocessing-distributed \
 --world-size 1 \
 --rank 0 \
 /imagenet"
