import torch

t1 = torch.Tensor([[1,2,3,4],[5,6,7,8]])
t2 = torch.unsqueeze(torch.Tensor([1,3]),1).type(torch.int64)
print(t2)
print(torch.gather(t1, 1, t2))