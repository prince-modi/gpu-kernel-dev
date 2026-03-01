import helion
import torch
import helion.language as hl
import os


def retrieve_configs(benchmark_name: str):
  filtered_configs = []
  if os.path.exists('configs'):
    all_configs = os.listdir('configs')
    for conf in all_configs:
        if benchmark_name in conf and 'json' in conf:
            filtered_configs.append(helion.Config.load(os.path.join('configs',conf)))
  return filtered_configs


config_list = retrieve_configs('helion_rms_kernel')


@helion.kernel(configs = config_list, static_shapes = False)
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