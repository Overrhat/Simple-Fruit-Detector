import os
import numpy as np
import tensorflow as tf
import tf2onnx

def ConvertTensorToOnnx(tf_model_path='classification_model.h5', onnx_model_path='classification_model.onnx'):
    model = tf.keras.models.load_model(tf_model_path)
    model.output_names = ['output']
    
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    #spec = (tf.TensorSpec((None, 128, 128, 3), tf.float32, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_model_path)
    return

# Convert the TensorFlow model to ONNX
ConvertTensorToOnnx('fruit_detector_model_224.keras', 'fruit_detector_model_224.onnx')