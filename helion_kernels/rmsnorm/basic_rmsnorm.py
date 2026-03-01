import helion
import torch
import helion.language as hl


@helion.kernel(
      helion.Config.load("configs/helion_rms_kernel-256x256x256.json"),
      helion.Config.load("configs/helion_rms_kernel-1024x1024x1024.json"),
      helion.Config.load("configs/helion_rms_kernel-8192x8192x8192.json")
)
def helion_rms_kernel(x: torch.Tensor, w: torch.Tensor, eps) -> torch.Tensor:
  out = torch.empty_like(x)
  m = x.shape[0]
  #dividing by the number of rows / vals in 0th axis
  for m_tile in hl.tile(m):
      row = x[m_tile,:].to(torch.float32) #prevents loss in precision
      sumsq = torch.rsqrt(torch.mean(row * row,dim = -1) + eps)
      row = (row * sumsq[:,None]) * w[:].to(torch.float32) #again prevents loss in precision
      out[m_tile,:] = row.to(out.dtype)
  return out