#!/bin/bash
#SBATCH --job-name resnet50
#SBATCH --nodes 1

srun --container-image $1 \
 --container-mounts $2:/imagenet \
 --no-container-entrypoint \
 /bin/bash -c \
 "python ./multiproc.py \
 --nproc_per_node $3 \
 ./launch.py \
 --model resnet50 \
 --precision $4 \
 --mode $5 \
 --platform $6 \
 /imagenet \
 --raport-file benchmark.json \
 --epochs 1 \
 --prof 100"
