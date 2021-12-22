# ResNet-50 on PyTorch

## Notes

This repository is a short summary of the Deep Learning Examples [repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) for running ResNet-50 on PyTorch. Additional information and instructions are available there.

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

### Training

#### Directly

Single and multi-GPU.

```bash
python ./multiproc.py --nproc_per_node <number of GPUs> ./launch.py --model resnet50 --precision <TF32|FP32|AMP> --mode benchmark_training --platform <DGX1V|DGX2V|DGXA100> /imagenet --raport-file benchmark.json --epochs 1 --prof 100
```

#### Slurm

Running from the login server requires to convert the Docker container into a squash file. This can be done using [Enroot](https://github.com/NVIDIA/enroot) by running:

```bash
enroot import dockerd://nvidia_resnet50_pt:latest
```

A `.sqsh` file will be created locally.

Single GPU:

1. Clone this repository:

    ```bash
    git clone https://gitlab.com/anahum/performance_tests
    cd ./ResNet-50/PyTorch
    ```

2. Submit a job for training:

    ```bash
    sbatch slurm_single_gpu.sub <path to .sqsh file> <path to resnet> <TF32|FP32|AMP> <DGX1V|DGX2V|DGXA100>
    ```

Multi-GPU and multi-node:

Due to issues with NVIDIA's code, this section is based on PyTorch [code](https://github.com/pytorch/examples/tree/master/imagenet).

1. Clone this repository as described above.
2. Clone PyTorch repository:

    ```bash
    git clone https://github.com/pytorch/examples
    cd ./imagenet
    ```

    `<path to pytorch code>`  is the directory in which the `main.py` file is placed.

3. By default it will run for 8 nodes where each have 8 GPUs. Edit `slurm_multi_gpu.sub` as needed.

4. Submit a job for training:

    ```bash
    sbatch slurm_multi_gpu.sub <path to .sqsh file> <path to pytorch code> <path to resnet>
    ```

### Inferencing

Single GPU only.

```bash
python ./launch.py --model resnet50 --precision <TF32|FP32|AMP> --mode benchmark_inference --platform <DGX1V|DGX2V|DGXA100> /imagenet --raport-file benchmark.json --epochs 1 --prof 100
```

### Expected results

Available [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5#results).
