# performance_tests

Training, inferencing and networking performance tests to run with chosen models on different hardware. Based on different sources such as NVIDIA's Deep Learning Examples GitHub repository.

# Quick PyTorch Docker benchmark test

Run the following commands to pull and run an NGC PyTorch 23.03 container equipped with the needed requirements to run a quick synthetic data benchmarking on a DGX A100 (or H100) with 8 GPUs, for 1 epoch and 100 steps:

1. Pull the container:

   ```bash
   docker pull assafna/benchmark:pytorch-23.03-py3
   ```

1. Run the benchmarking:

   ```bash
   docker run --rm --gpus 8 assafna/benchmark:pytorch-23.03-py3 \
   python ./DeepLearningExamples/PyTorch/Classification/ConvNets/multiproc.py \
   --nproc_per_node 8 \
   ./DeepLearningExamples/PyTorch/Classification/ConvNets/launch.py \
   --model resnet50 \
   --precision AMP \
   --mode benchmark_training \
   --platform DGXA100 \
   --data-backend synthetic \
   --raport-file benchmark.json \
   --epochs 1 \
   --prof 100 \
   ./
   ```
