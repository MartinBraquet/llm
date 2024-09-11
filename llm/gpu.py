import torch
import torch_directml

device = torch_directml.device()

tensor1 = torch.tensor([1]).to(device)
tensor2 = torch.tensor([2]).to(device)
dml_algebra = tensor1 + tensor2
print(dml_algebra)