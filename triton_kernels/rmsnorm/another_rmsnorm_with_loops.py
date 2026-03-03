import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel_3(output_ptr,input_ptr,w_ptr,eps,n_row,input_row_stride,ROW_INTERVAL:tl.constexpr, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr, warp_specialize: tl.constexpr):
    # Each program handles one row
    row = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_off in range(row,n_row,ROW_INTERVAL):
       X_row = input_ptr + row_off * input_row_stride
       Y_row = output_ptr + row_off * input_row_stride
       sum_sq = tl.zeros([BLOCK_SIZE],dtype=tl.float32)
       for off in tl.range(0, input_row_stride, BLOCK_SIZE):
           cols = off + tl.arange(0, BLOCK_SIZE)
           mask = cols < input_row_stride
           x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
           sum_sq += x * x
       mean_sq = (tl.sum(sum_sq) / input_row_stride).to(tl.float32)
       normfactor = 1.0 / tl.sqrt(mean_sq + eps)
       for off in tl.range(0,input_row_stride,BLOCK_SIZE,num_stages = num_stages):
           cols = off + tl.arange(0,BLOCK_SIZE)
           mask = cols < input_row_stride
           x_block = tl.load(X_row + cols, mask = mask, other = 0.0).to(tl.float32)
           w_block = tl.load(w_ptr + cols, mask=mask,other = 0.0).to(tl.float32)
           tl.store(Y_row + cols, (x_block * w_block * normfactor).to(tl.float16), mask=mask)
