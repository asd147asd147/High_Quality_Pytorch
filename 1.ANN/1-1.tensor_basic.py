import torch

x = torch.tensor([[1,2,3],[4,5,6,],[7,8,9]])
print(x)
print("Size: ", x.size())
print("Shape: ", x.shape)
print("랭크(차원): ", x.ndimension())

x = torch.unsqueeze(x,0)
print(x)
print("Size: ", x.size())
print("Shape: ", x.shape)
print("랭크(차원): ", x.ndimension())

x = torch.squeeze(x)
print(x)
print("Size: ", x.size())
print("Shape: ", x.shape)
print("랭크(차원): ", x.ndimension())

x = x.view(9)
print(x)
print("Size: ", x.size())
print("Shape: ", x.shape)
print("랭크(차원): ", x.ndimension())

try:
    x = x.view(2,4)
except Exception as e:
    print(e)