import triton
import triton.language as tl

#program ver based on Dario and https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/#fusion-the-real-win
@triton.jit
def rmsnorm_kernel_1(output_ptr,input_ptr,w_ptr,eps,input_row_stride,BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    # Each program handles one row
    row = tl.program_id(0)

    # Pointers for this row
    X_row = input_ptr + row * input_row_stride
    Y_row = output_ptr + row * input_row_stride

    # Accumulate sum of squares in registers (FP32 for precision)
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

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
