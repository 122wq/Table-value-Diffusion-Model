import torch
import torch.nn as nn
import onnx
from Model import ConditionalDiffusion_CycleGAN_tequan_Model

save_dict = torch.load("Fold0Weights_original.pkl", map_location= "cpu", weights_only= True)
model = ConditionalDiffusion_CycleGAN_tequan_Model(input_dim= 7, output_dim=2)
model.load_state_dict(save_dict)
model.eval()
#print(model(torch.randn(1,2), torch.tensor([[155, 93.78, 123.394, 26.324, 1, 1, 88]]), torch.tensor([500]), False))
inputs = (torch.randn(1,2),torch.randn(1,7), torch.tensor([10.0]), False)

onnx_program = torch.onnx.export(
    model,
    inputs,
    "onnx_diffusion.onnx",
    input_names=["noise", "cond", "t", "train_flag"],
    output_names=["output_fake", "validity_fake"],
    opset_version=17,
)

onnx_model  =onnx.load("onnx_diffusion.onnx")
onnx.checker.check_model(onnx_model)
