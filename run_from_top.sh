#atm only have helion for rmsnorm
python3 helion_kernels/rmsnorm/generate_config.py
python3 helion_kernels/flash_attn/generate_config.py

python3 bench-driver.py

rms_kernels=("rms_bench-helion-helion_rms_kernel","rms_bench-torch-rmsnorm","rms_bench-triton-rmsnorm_with_loops","rms_bench-cute-cute_rms_norm") 
attn_kernels=("attn_bench-torch-flash_attn", "attn_bench-helion-flashatt_fwd", "attn_bench-triton-forward","attn_bench-cute-flashatt_fwd")

./generate_dumps.sh nsys ${rms_kernels[*]}
./generate_dumps ncu ${rms_kernels[*]}
./generate_dumps.sh nsys ${attn_kernels[*]}
./generate_dumps ncu ${attn_kernels[*]}

