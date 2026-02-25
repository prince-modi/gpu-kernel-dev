import torch

def rmsnorm_kernel_basic(X:torch.Tensor, W: torch.Tensor, eps):
  x2_mean = torch.mean(X * X, dim = -1,keepdim=True)
  divided_sums = torch.rsqrt(x2_mean + eps)
  return (X * divided_sums * W).to(dtype=X.dtype)
