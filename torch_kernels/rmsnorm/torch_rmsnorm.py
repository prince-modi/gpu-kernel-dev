import torch

def rmsnorm_kernel_basic(X:torch.Tensor, W: torch.Tensor, eps):
  x_stride = X.stride(0) # n for mean calc
  x2_sums = torch.sum(X * X, dim = 0)
  divided_sums = torch.rsqrt(x2_sums / x_stride + eps)
  print(X.shape)
  print(divided_sums.shape)
  print(W.shape)
  return (X * divided_sums * W[:]).to(dtype=X.dtype)
