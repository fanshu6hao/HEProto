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

SEEDS=(171 354) # 171 354 550 667 985

N=5  # 5 10
K=1   # 1 5

mode=intra
gpu_device=0

batch_size=16

lr_span=2e-05
lr_type=2e-05

# train 
for seed in ${SEEDS[@]}; do
    python3 main.py \
        --seed=${seed} \
        --gpu_device=${gpu_device} \
        --mode=${mode} \
        --N=${N} \
        --K=${K} \
        --lr_span=${lr_span} \
        --lr_type=${lr_type} \
        --ft_lr_span=${lr_span} \
        --ft_lr_type=${lr_type} \
        --eval_every_steps=100 \
        --name=MT \
        --batch_size=${batch_size} \
        --max_training_steps=1000 \
        --load_best_and_test=True \
        --ignore_eval_test
done      