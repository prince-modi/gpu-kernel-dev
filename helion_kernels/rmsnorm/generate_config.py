import os

import torch
import helion

from helion_kernels.rmsnorm.basic_rmsnorm import helion_rms_kernel
from helion_kernels.generate_config import DEVICE,autotune

rmsnorm_configs = [ 
        (
            #small config
            torch.randn((256, 256), dtype= torch.float16, device=DEVICE),
            torch.randn((256,), dtype= torch.float16, device=DEVICE),
            1e-5,
        )
        (
            #medium config
            torch.randn((1024,1024), dtype= torch.float16,device=DEVICE),
            torch.randn((1024,), dtype= torch.float16,device = DEVICE),
            1e-5
        ),
        (
            #large config
            torch.randn((8192,8192), dtype= torch.float16, device=DEVICE),
            torch.randn((8192,), dtype= torch.float16,device = DEVICE),
            1e-5
        )
]


if __name__ == "__main__":
    if(os.path.exists('configs') == False):
        os.mkdir('configs')
    #rmsnorm_config_block
    rmsnorm_functions = [helion_rms_kernel]
    for func in rmsnorm_functions:
        autotune(func, rmsnorm_configs,force_autotune=True)
            
    
    #attention_config_block
    # attention_functions = []
    # for func in attention_functions:
    #     for config in attention_configs:
    #         autotune(func,config)
