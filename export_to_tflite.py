import numpy
import os
import shutil
import subprocess

import onnx
from onnx_tf.backend import prepare
import torch
import torch.onnx
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from models import ContactTracingTransformer
from loader import get_dataloader

NB_EXAMPLES_FOR_SANITY_CHECK=5

# Load pytorch model
pytorch_model = ContactTracingTransformer()
pytorch_model.load_state_dict(torch.load("models/model.pth"))
pytorch_model.eval()

# Load dataset (used for sanity checking the converted models)
path = "./data/"
dataloader = get_dataloader(batch_size=1, shuffle=False, num_workers=0, path=path)
batch = next(iter(dataloader))

# Get list of inputs names as in the batch
input_names=[]
for i in batch:
    input_names.append(i)
output_names = ['encounter_variables', 'latent_variable']

print(input_names)
import pdb; pdb.set_trace()
# Convert PyTorch model to ONNX format 
torch.onnx.export(pytorch_model,            
                  batch,                         
                  "model_onnx_10.onnx",   
                  export_params=True,        
                  opset_version=10,          
                  do_constant_folding=True,  
                  input_names=input_names,  
                  output_names=output_names,
                  dynamic_axes={
                    'mask' : {0: 'batch_size', 1: 'sequence'},
                    'health_history' : {0: 'batch_size'},
                    'health_profile' : {0: 'batch_size'},
                    'infectiousness_history' : {0: 'batch_size'},
                    'history_days' : {0: 'batch_size'},
                    'current_compartment' : {0: 'batch_size'},
                    'encounter_health' : {0: 'batch_size', 1: 'sequence'},
                    'encounter_message' : {0: 'batch_size', 1: 'sequence'},
                    'encounter_day' : {0: 'batch_size', 1: 'sequence'},
                    'encounter_partner_id' : {0: 'batch_size', 1: 'sequence'},
                    'encounter_duration' : {0: 'batch_size', 1: 'sequence'},
                    'encounter_is_contagion' : {0: 'batch_size', 1: 'sequence'}
                  }) 
                  
# Load ONNX model and convert to TF model
onnx_model = onnx.load("model_onnx_10.onnx")  
tf_model = prepare(onnx_model)

# Sanity-check the TF model
deltas = []
for i, batch in enumerate(iter(dataloader)):
    pytorch_output = pytorch_model(batch)
    tf_output = tf_model.run(batch)
    
    for k in pytorch_output.keys():
        k_delta = pytorch_output[k].detach().numpy() - getattr(tf_output, k)
        deltas.append(k_delta)
    
    if i >= NB_EXAMPLES_FOR_SANITY_CHECK:
        break # Limit the testing to avoid spending too much time on it.

abs_deltas = numpy.abs(numpy.hstack(deltas))
print("Tensorflow Model sanity check")
print("Min abs. diff with pytorch model output : %f" % abs_deltas.min())
print("Mean abs. diff with pytorch model output : %f" % abs_deltas.mean())
print("Max abs. diff with pytorch model output : %f" % abs_deltas.max())

# Export TF model as frozen inference graph
tf_model.export_graph('tf_model.pb')

# Convert the tf graph to a TF Saved Model
save_dir = "./tmp_tfmodel_conversion/"
if os.path.isdir(save_dir):
    print('Already saved a model, cleaning up')
    shutil.rmtree(save_dir, ignore_errors=True)

builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(save_dir)
with tf.compat.v1.Session(graph=tf_model.graph) as sess:
    
    input_spec = {}
    output_spec = {}
    for name in tf_model.inputs:
        input_spec[name] = tf_model.tensor_dict[name]
    for name in output_names:
        output_spec[name] = tf_model.tensor_dict[name]
    
    sigs = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY :
            tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(input_spec, output_spec)}
            
    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)
    builder.save()


# Convert Saved Model to TFLite model
converter = tf.lite.TFLiteConverter.from_saved_model(save_dir)
import pdb; pdb.set_trace()
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)

# Sanity-check the TFLite model

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
import pdb; pdb.set_trace()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

import pdb; pdb.set_trace()

