#need to debug why autotuner is not working when compiling in triton
@helion.kernel(autotune_effort = "none")
def helion_rms_kernel(x: Tensor, w: Tensor, eps) -> Tensor:
  out = torch.empty_like(x)
  m = x.shape[0]
  #dividing by the number of rows / vals in 0th axis
  for m_tile in hl.tile(m):
      row = x[m_tile,:]
      sumsq = torch.rsqrt(torch.mean(row * row,dim = -1) + eps)
      row = (row * sumsq[:,None]) * w[:]
      out[m_tile,:] = row
  return out