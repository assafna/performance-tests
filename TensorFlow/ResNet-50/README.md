# ResNet-50 on TensorFlow

## Notes

This repository is a short summary of the Deep Learning Examples [repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5) for running ResNet-50 on TensorFlow. Expected results are available [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#results).

## Requirements

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
* [PyTorch 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-tensorflow) or newer (will be downloaded automatically when building the container).
* Supported GPUs:
  * [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/).
  * [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/).
  * [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/).

## Instructions

### Build the container

1. Clone the repository:

    ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd ./DeepLearningExamples/TensorFlow/Classification/ConvNets
    ```

2. Build the ResNet-50 PyTorch NGC container.

    ```bash
    docker build --network=host . -t nvidia_resnet50_tf
    ```

### Download and preprocess the dataset

1. Sign-up or login to [ImageNet](https://image-net.org/).
2. Download the [ILSVRC2012 images dataset](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php):

    * Training images (Task 1 & 2).
    * Validation images (all tasks).
    * Training bounding box annotations (Task 1 & 2 only).

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

5. Extract the bounding boxes:

    ```bash
    mkdir bounding_boxes && mv ILSVRC2012_bbox_train_v2.tar.gz bounding_boxes/ && cd bounding_boxes
    tar -xvf ILSVRC2012_bbox_train_v2.tar.gz && rm -f ILSVRC2012_bbox_train_v2.tar.gz
    cd ..
    ```

6. Convert the dataset to TFRecords:

    * Run the TensorFlow container:

        ```bash
        docker run --rm -it --name=rn50_tf -v <path to imagenet>:/imagenet --ipc=host nvidia_resnet50_tf
        ```

        `<path to imagenet>`  is the directory in which the `train/` and `val/` directories are placed.

    * Run the script:

        ```bash
        mkdir /imagenet/result && ./dataprep/preprocess_imagenet.sh /imagenet
        ```

7. Optional: Create DALI index files:

```bash
bash ./utils/dali_index.sh /imagenet/result /imagenet/dali_idx
```

### [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5)

When running multi-node `<node index>` should be set to `0` for the "master" node and increased for any additional node (i.e., 1, 2, 3...).

Optional arguments to be added:

* `--amp` - Automated Mixed Precision.
* `--xla` - Accelerated Linear Algebra.
* `--dali` - DALI, together with `--data_idx_dir`.

Additional parameters can be found [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#parameters).

__Single node:__

```bash
mpiexec --allow-run-as-root \
--bind-to socket \
-np <number of GPUs> \
python ./main.py \
--mode <training_benchmark|inference_benchmark> \
--batch_size 256 \
--data_dir /imagenet/result \
--results_dir /results
```

__Multi-node:__

Not yet was able to run.

### Slurm

Running from the login server requires to convert the Docker container into a squash file. This can be done using [Enroot](https://github.com/NVIDIA/enroot) by running:

```bash
enroot import dockerd://nvidia_resnet50_tf:latest
```

A `.sqsh` file will be created locally.

Clone this repository:

```bash
git clone https://gitlab.com/anahum/performance_tests
cd ./performance_tests/TensorFlow/ResNet-50/Slurm
```

#### NVIDIA DeepLearningExamples

__Single node:__

```bash
sbatch \
--gres gpu:<number of GPUs> \
./DeepLearningExamples/single_node.sub \
<path to .sqsh file> \
<path to resnet> \
<number of GPUs> \
<training_benchmark|inference_benchmark>
```

__Multi node:__

```bash
sbatch \
--nodes <number of nodes> \
--gres gpu:<number of GPUs per node> \
--ntasks-per-node <number of GPUs per node> \
./DeepLearningExamples/multi_node.sub \
<path to .sqsh file> \
<path to resnet> \
<total number of GPUs> \
<training_benchmark|inference_benchmark>
```
