#!/bin/bash
#c=D
d=0
bs=512
data_size=100000
epochs=80
eval_interval=10

cs=( T W )
for c in ${cs[@]}; do
    echo $c
    CUDA_VISIBLE_DEVICES=$d python matcher.py \
                --char $c \
                --synthetic_data_dir /projects/nvog/synthetic/data/synthetic/shakespeare_2021-09-28/${c}_twin_20_2021-10-08 \
                --model_type Attention \
                --batch_size $bs \
                --megabatch_size 4 \
                --negative_mining UniformSampling \
                --n_epochs $epochs \
                --softmax_temperature 1.0 \
                --eval_interval $eval_interval \
                --optimizer Adam \
                --lr 0.001 \
                --conv_template_residual \
                --collapse_attn_operation hw_sum \
                --limit_dataset_size $data_size \
                --jitter_triplet &> /dev/null &
    d=$((d+1))
done
