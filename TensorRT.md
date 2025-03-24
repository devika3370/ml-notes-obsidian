# TensorRT

Resources:
- https://developer.nvidia.com/tensorrt
- https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html#work
- https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html

NVIDIA TensorRT is a high-performance deep learning "inference optimizer" and runtime that accelerates inference on NVIDIA GPUs. It takes a trained model (a model in pytorch, tensorflow or onnx) and optimizes it by:
1) Layer Fusion: Combining operations to reduce latency
2) Precision Calibration: Converting models to FP16 or INT8 for faster inference
3) Kernel Auto-Tuning: Selecting the best kernel for the hardware. 

```
def load_serialized_engine(filename):

    print("Loading deserialized TRT engine.")

    with open(filename, 'rb') as engine_file:

        return engine_file.read()
```