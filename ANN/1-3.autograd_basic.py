import torch

w = torch.tensor(1.0, requires_grad=True)
a = w*3
l = a**2

l.backward()
print('dl/dw={}'.format(w.grad))