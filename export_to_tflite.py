import numpy
import onnx
from onnx_tf.backend import prepare
import torch
import torch.onnx

from models import ContactTracingTransformer
from loader import get_dataloader

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
 
# Convert PyTorch model to ONNX format 
torch.onnx.export(pytorch_model,            
                  batch,                         
                  "model_onnx_10.onnx",   
                  export_params=True,        
                  opset_version=10,          
                  do_constant_folding=True,  
                  input_names=input_names,  
                  output_names=output_names) 

                  
# Load ONNX model and convert to TF model
onnx_model = onnx.load("model_onnx_10.onnx")  
tf_model = prepare(onnx_model)

# Sanity-check the TF model
deltas = None
for i, batch in enumerate(iter(dataloader)):
    print([x.shape for x in batch.values()])
    pytorch_output = pytorch_model(batch)
    tf_output = tf_model.run(batch)
    
    for k in pytorch_output.keys():
        k_delta = pytorch_output[k].detach().numpy() - getattr(tf_output, k)
        if deltas is None:
            deltas = k_delta
        else:
            deltas = numpy.hstack((deltas, k_delta))
    print(deltas.size)
    
print("Tensorflow Model sanity check")
print("Min diff with pytorch model output : %f" % deltas.min())
print("Mean diff with pytorch model output : %f" % deltas.mean())
print("Max diff with pytorch model output : %f" % deltas.max())

# Export TF model as frozen inference graph
tf_model.export_graph('tf_graph2.pb')