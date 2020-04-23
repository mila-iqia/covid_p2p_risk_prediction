import torch
from models import ContactTracingTransformer
import torch.onnx
from loader import get_dataloader

model = ContactTracingTransformer()
model.load_state_dict(torch.load("models/model.pth"))
model.eval()
path = "output.pkl"
dataloader = get_dataloader(batch_size=1, shuffle=False, num_workers=0, path=path)
batch = next(iter(dataloader))
#List of inputs as in the batch
input_names=[]
for i in batch:
    input_names.append(i)
torch.onnx.export(model,            
                  batch,                         
                  "model_onnx.onnx",   
                  export_params=True,        
                  opset_version=9,          
                  do_constant_folding=True,  
                  input_names = input_names,  
                  output_names = ['latent_variable','encounter_variable']) 