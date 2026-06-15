d=0
max_gpu=3
# A B C D E F G H K L M N P R T W
char=A
batch_size=64
nepochs=121
evalfreq=10
margin=0.3
lr=0.0001
attn_op="hw_sum"  # filter_max  currently ran hw_sum but let's change
model_type=Attention  # Attention
metric="syn_valid"
lds=5000

#
# best so far with hw_sum was 2 ncb and 4 cpb
#  /trunk/nvog/matching_results_aaai2022_2/logs/CharA_TempResidual_ModelAttention_2CNNBlocks_4ConvPerBlock_EmbSz128_hw_sum_Temp1.0_Margin0.3_Bsz64_AdamLR0.0001-2022-08-04_21:37:37/log.log
# best_syn_valid_recall_pct_A.5': 0.555
# best_gt_test_pos_recall_pct_A.5': 0.5511811023622047
# best_gt_test_strong_neg_recall_pct_A.5': 0.5098425196850394
# best_gt_test_weak_neg_recall_pct_A.5': 0.4921259842519685
#
#
#
#

# d=0
# ncb=2
# ncpb=3
# for smt in 0.2 0.4 0.6 0.8; do
#     CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
#             --char $char \
#             --margin $margin \
#             --model_type $model_type \
#             --batch_size $batch_size \
#             --n_epochs $nepochs \
#             --softmax_temperature $smt \
#             --eval_interval $evalfreq \
#             --optimizer Adam \
#             --lr $lr \
#             --conv_template_residual \
#             --collapse_attn_operation $attn_op \
#             --num_cnn_blocks $ncb \
#             --num_conv_per_block $ncpb \
#             --stopping_metric $metric \
#             --wandb &
#             #--jitter_triplet \
#             #--conv_template_residual \
#             #--l2_baseline
#     d=$((d+1))
#     #if [ "$d" -gt "$max_gpu" ]; then
#     #    d=0
#     #fi
# done

# wait
# d=0

ncb=2   # 2 blocks total for attention
attn_op="hw_sum"
for ncpb in 2 3 4 5; do
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
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

exit

ncb=2   # 2 blocks total for attention
attn_op="hw_max"  
for ncpb in 3 4 5 6; do
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
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#           L2 Embedding
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


model_type=L2Embedding
d=0
ncb=2   # 2 blocks total for attention
for ncpb in 3 4 5 6; do
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
            --conv_template_residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

d=0
ncb=3   # 2 blocks total for attention
for ncpb in 3 4 5 6; do
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
            --conv_template_residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

d=0
ncb=4   # 2 blocks total for attention
for ncpb in 3 4 5 6; do
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
            --conv_template_residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

