# Performance Tests

Training, inferencing and networking performance tests to run with chosen models on different hardware. Based on different sources such as NVIDIA's Deep Learning Examples GitHub repository.

# Quick PyTorch Docker / K8S benchmark test

Run the following command to pull and run an NGC PyTorch container equipped with the needed requirements to run a quick synthetic data benchmarking on a DGX A100 (or H100) with 8 GPUs, for 1 epoch and 100 steps:

* Docker:
   ```bash
   docker run --rm --gpus 8 assafna/benchmark:pytorch-<23.03 or 23.10>-py3 \
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

* Kubernetes:
   ```yaml
   apiVersion: batch/v1
   kind: Job
   metadata:
     name: gpu-benchmark
     namespace: <relevant namespace>
   spec:
     selector: {}
     template:
       metadata:
         name: gpu-benchmark
       spec:
         containers:
           - name: gpu-benchmark
             image: assafna/benchmark:pytorch-<23.03 or 23.10>-py3
             command:
               - python
               - './DeepLearningExamples/PyTorch/Classification/ConvNets/multiproc.py'
               - '--nproc_per_node=8'
               - './DeepLearningExamples/PyTorch/Classification/ConvNets/launch.py'
               - '--model=resnet50'
               - '--precision=AMP'
               - '--mode=benchmark_training'
               - '--platform=DGXA100'
               - '--data-backend=synthetic'
               - '--raport-file=benchmark.json'
               - '--epochs=1'
               - '--prof=100'
               - './'
         restartPolicy: OnFailure
   ```
