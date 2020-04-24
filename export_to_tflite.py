import numpy
import subprocess

import onnx
from onnx_tf.backend import prepare
import torch
import torch.onnx

from models import ContactTracingTransformer
from loader import get_dataloader

NB_EXAMPLES_FOR_SANITY_CHECK=500

# Load pytorch model
pytorch_model = ContactTracingTransformer()
pytorch_model.load_state_dict(torch.load("models/model.pth"))
pytorch_model.eval()

# Load dataset (used for sanity checking the converted models)
path = "output.pkl"
dataloader = get_dataloader(batch_size=1, shuffle=False, num_workers=0, path=path)
batch = next(iter(dataloader))

# Get list of inputs names as in the batch
input_names=[]
for i in batch:
    input_names.append(i)
output_names = ['encounter_variables', 'latent_variable']

print(input_names)
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
                    'mask' : {1: 'sequence'},
                    'encounter_health' : {1: 'sequence'},
                    'encounter_message' : {1: 'sequence'},
                    'encounter_day' : {1: 'sequence'},
                    'encounter_partner_id' : {1: 'sequence'}
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

# Convert the inference graph to TFLite
tflite_template = "tflite_convert --graph_def_file tf_model.pb --output_file model.tflite --output_format TFLITE --input_arrays %s --output_arrays %s --allow_custom_ops"
tflite_command = tflite_template % (','.join(input_names), ','.join(output_names))
subprocess.run(tflite_command, shell=True)
import pdb; pdb.set_trace()

#tflite_convert --graph_def_file tf_model.pb --output_file model.tflite --output_format TFLITE --input_arrays health_history,history_days,encounter_health,encounter_message,encounter_day,encounter_partner_id,mask --output_arrays 'latent_variable','encounter_variable' --allow_custom_ops