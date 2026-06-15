
d=0  #""
max_gpu=3
# A B C D E F G H K L M N P R T W
char=K
batch_size=64
nepochs=101
evalfreq=10
margin=0.3
lr=0.0001
attn_op="hw_sum"
model_type=StackedBCE
lds=5000
ncb=4
ncpb=3

#CUDA_VISIBLE_DEVICES=$d python3 -m pdb matcher.py \
for char in G; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature 1.0 \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --collapse_attn_operation $attn_op \
            --limit_dataset_size $lds \
            --output_dir /trunk/nvog/matching_results_aaai2022_debug \
            --log_dir /trunk/nvog/matching_results_aaai2022_debug/logs
            #--conv_template_residual \
            #--jitter_triplet \
            #--wandb
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done
