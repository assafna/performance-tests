# ResNet-50 on PyTorch

## Notes

This repository is a short summary of the Deep Learning Examples [repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) for running ResNet-50 on PyTorch. Expected results are available [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5#results).

## Model overview

The ResNet50 v1.5 model is a modified version of the [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385).

The model is initialized as described in [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://arxiv.org/pdf/1502.01852.pdf).

## Requirements

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
* [PyTorch 21.03-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) or newer (will be downloaded automatically when building the container).
* Supported GPUs:
  * [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/).
  * [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/).
  * [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/).

## Instructions

### Download and preprocess the dataset

1. Sign-up or login to [ImageNet](https://image-net.org/).
2. Download the [ILSVRC2012 images dataset](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) - Training images (Task 1 & 2) & Validation images (all tasks).
3. Extract the training data:

    ```bash
    mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
    ```

4. Extract the validation data and move the images to subfolders:

    ```bash
    mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    ```

### Build and run a container

1. Clone the repository:

    ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd ./DeepLearningExamples/PyTorch/Classification/ConvNets
    ```

2. Build the ResNet-50 PyTorch NGC container.

    ```bash
    docker build --network=host . -t nvidia_resnet50_pt
    ```

3. Run the container.

    ```bash
    docker run --rm -it --name=nvidia_resnet50_pt -v <path to imagenet>:/imagenet --ipc=host nvidia_resnet50_pt
    ```

    `<path to imagenet>`  is the directory in which the `train/` and `val/` directories are placed.

### [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)

When running multi-node `<node index>` should be set to `0` for the "master" node and increased for any additional node (i.e., 1, 2, 3...).

#### Single node

```bash
python ./multiproc.py \
--nproc_per_node <number of GPUs> \
./launch.py \
--model resnet50 \
--precision <TF32|FP32|AMP> \
--mode <benchmark_training|benchmark_inference> \
--platform <DGX1V|DGX2V|DGXA100> \
/imagenet \
--raport-file benchmark.json \
--epochs 1 \
--prof 100
```

#### Multi-node

Run on each node:

```bash
python ./multiproc.py \
--nnodes <number of nodes> \
--node_rank <node index> \
--nproc_per_node <number of GPUs per node> \
--master_addr <master node address> \
--master_port <a free port> \
./launch.py \
--model resnet50 \
--precision <TF32|FP32|AMP> \
--mode <benchmark_training|benchmark_inference> \
--platform <DGX1V|DGX2V|DGXA100> \
/imagenet \
--raport-file benchmark.json \
--epochs 1 \
--prof 100
```

### [NVIDIA Apex](https://github.com/NVIDIA/apex/tree/master/examples/imagenet) (mixed precision, DDP)

#### Single node

```bash
python -m torch.distributed.launch \
--nproc_per_node <number of GPUs> \
./examples/apex/examples/imagenet/main_amp.py -a resnet50 \
--opt-level O1 \
/imagenet
```

#### Multi-node

Run on each node:

```bash
python -m torch.distributed.launch \
--nnodes <number of nodes> \
--node_rank <node index> \
--nproc_per_node <number of GPUs per node> \
--master_addr <master node address> \
--master_port <a free port> \
./examples/apex/examples/imagenet/ -a resnet50 \
--opt-level O1 \
/imagenet
```

### [PyTorch example](https://github.com/pytorch/examples/tree/master/imagenet) (DDP)

Clone the repository:

```bash
git clone https://github.com/pytorch/examples
cd ./examples/imagenet
```

#### Single node

```bash
python ./main.py \
-a resnet50 \
--dist-url 'tcp://127.0.0.1:<a free port>' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
/imagenet
```

#### Multi-node

Run on each node:

```bash
python ./main.py \
-a resnet50 \
--dist-url 'tcp://<master node address>:<a free port>' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size <number of nodes> \
--rank <node index> \
/imagenet
```

### Slurm

Running from the login server requires to convert the Docker container into a squash file. This can be done using [Enroot](https://github.com/NVIDIA/enroot) by running:

```bash
enroot import dockerd://nvidia_resnet50_pt:latest
```

A `.sqsh` file will be created locally.

Clone this repository:

```bash
git clone https://gitlab.com/anahum/performance_tests
cd ./performance_tests/ResNet-50/PyTorch/Slurm
```

#### NVIDIA DeepLearningExamples

Single node:

```bash
sbatch \
--gres gpu:<number of GPUs> \
./DeepLearningExamples/single_node.sub \
<path to .sqsh file> \
<path to resnet> \
<number of GPUs> \
<TF32|FP32|AMP> \
<benchmark_training|benchmark_inference> \
<DGX1V|DGX2V|DGXA100>
```

Multi-node:

```bash
sbatch \
--nodes <number of nodes> \
--gres gpu:<number of GPUs per node> \
./DeepLearningExamples/multi_node.sub \
<path to .sqsh file> \
<path to resnet> \
<number of nodes> \
<number of GPUs per node> \
<a free port> \
<TF32|FP32|AMP> \
<benchmark_training|benchmark_inference> \
<DGX1V|DGX2V|DGXA100>
```

#### NVIDIA Apex (mixed precision, DDP)

Modify the `.sub.` files to change parameters such as the optimization level.

Single node:

```bash
sbatch \
--gres gpu:<number of GPUs> \
./Apex/single_node.sub \
<path to .sqsh file> \
<path to resnet> \
<number of GPUs>
```

Multi-node:

```bash
sbatch \
--nodes <number of nodes> \
--gres gpu:<number of GPUs per node> \
./Apex/multi_node.sub \
<path to .sqsh file> \
<path to resnet> \
<number of nodes> \
<number of GPUs per node> \
<a free port>
```

#### PyTorch example (DDP)

Clone the repository:

```bash
git clone https://github.com/pytorch/examples
```

`<path to pytorch>` will refer to this repository path.

Single node:

```bash
sbatch \
--gres gpu:<number of GPUs> \
./PyTorch/single_node.sub \
<path to .sqsh file> \
<path to resnet> \
<path to pytorch> \
<a free port>
```

Multi node:

```bash
sbatch \
--nodes <number of nodes> \
--gres gpu:<number of GPUs per node> \
./PyTorch/multi_node.sub \
<path to .sqsh file> \
<path to resnet> \
<path to pytorch> \
<a free port> \
<number of nodes>
```
