def rmsnorm(X:torch.Tensor, W: torch.Tensor, eps):
  x_stride = X.stride(0) # n for mean calc
  x2_sums = torch.sum(X * X, dim = 0)
  divided_sums = torch.rsqrt(torch.mean(X * X,dim = -1) + eps)
  return (X * divided_sums * W[:]).to(dtype=X.dtype)
