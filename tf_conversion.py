import onnx
from onnx_tf.backend import prepare
from models import ContactTracingTransformer
from loader import get_dataloader
import torch
path = "output.pkl"
dataloader = get_dataloader(batch_size=1, shuffle=False, num_workers=0, path=path)
batch = next(iter(dataloader))

#Load ONNX model
onnx_model = onnx.load("model_onnx_10.onnx")  
tf_model = prepare(onnx_model)
#Inputs to the model
print('inputs:', tf_model.inputs)
# Output nodes from the model
print('outputs:', tf_model.outputs)

# All nodes in the model
# print('tensor_dict:')
# print(tf_model.tensor_dict)  
output=tf_model.run(batch)
print(output)
tf_model.export_graph('tf_graph2.pb')

#Sanity check with the PyTorch Model
ctt = ContactTracingTransformer(pool_latent_entities=False, use_logit_sink=False)
ctt.load_state_dict(torch.load('model.pth'))
ctt.eval()
output = ctt(batch)
print(output)
