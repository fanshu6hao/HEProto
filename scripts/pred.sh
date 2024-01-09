#   Mode        batch_size      lr_span     lr_type
#                        INTER
#   5-1         16              1e-05       1e-05
#   10-1        8               1e-05       1e-05
#   5-5         4               1e-05       2e-05
#   10-5        2               1e-05       3e-05
#
#                        INTRA
#   5-1         16              2e-05       2e-05
#   10-1        8               2e-05       2e-05
#   5-5         4               2e-05       3e-05
#   10-5        2               2e-05       3e-05

seed=171
N=10  # 5 10 
K=1   # 1 5

mode=inter # inter or intra
gpu_device=1
max_training_steps=1000

# mt_model_dir 是训练好的模型地址，注意对应好任务设置
mt_model_dir=outputs/models-10-1-inter/bs_8-lr_1e-05_1e-05-steps_1000-seed_171-MT_add_auto_aw0.1_clw0.1_token0.1_gate_cw1.0_fmw0.1_m4.0

ft_lr_span=1e-05
ft_lr_type=1e-05

# train
python3 main.py \
    --seed=${seed} \
    --gpu_device=${gpu_device} \
    --mode=${mode} \
    --N=${N} \
    --K=${K} \
    --ft_lr_span=${ft_lr_span} \
    --ft_lr_type=${ft_lr_type} \
    --max_training_steps=${max_training_steps} \
    --load_best_and_test=True \
    --ignore_eval_test \
    --mt_test_only \
    --mt_model_dir=${mt_model_dir}
