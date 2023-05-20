import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set the GPU devices you want to use (e.g., "0,1,2")

# Create a TensorFlow session
sess = tf.Session()

# Check if GPU support is available
gpu_available = tf.test.is_gpu_available()

if gpu_available:
    print("GPU is available with TensorFlow.")
else:
    print("GPU is not available with TensorFlow.")
