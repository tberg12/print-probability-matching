
d=0
max_gpu=3
# A B C D E F G H K L M N P R T W
char=A
batch_size=64
nepochs=151
evalfreq=10
margin=0.3
lr=0.0001
attn_op="filter_sum"
model_type=Attention # Attention
ncb=2
ncpb=3
smt=1.0
metric="syn_valid"


residual="--conv_template_residual"
for char in G; do  #C L P; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --evaluate_ckpt \
            --evaluate_ckpt_dest /trunk/nvog/matching_results_aaai2022_eval \
            --load_model "/trunk/nvog/matching_results_aaai2022_SunAug14/CharG_TempResidual_ModelAttention_5000TrainPairs_GlobalInkTrue_2CNNBlocks_3ConvPerBlock_EmbSz128_hw_sum_Temp1.0_Jitter_Margin0.3_Bsz64_AdamLR0.0001-2022-08-14_21:10:24/epoch60.pt"
            #--wandb &
            #--load_model "/trunk/nvog/matching_results_aaai2022_SatAug6/CharA_TempResidual_ModelAttention_2CNNBlocks_3ConvPerBlock_EmbSz128_filter_sum_Temp1.0_Margin0.3_Bsz64_AdamLR0.0001-2022-08-06_03:06:27/best.pt"
            #--rerank_with_l2 \
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0
# TODO TODO TODO TODO TODO
exit

for char in T F M G; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in B D H W; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in R E N K; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done


wait
d=0

# repeat but with different settings
residual="--input_residual"

for char in A C L P; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in T F M G; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in B D H W; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in R E N K; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0


#######
######    EMBEDDING
######
######

model_type=L2Embedding  # Attention
ncb=4
ncpb=3

residual="--conv_template_residual"
for char in A C L P; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in T F M G; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in B D H W; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in R E N K; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done


wait
d=0

# repeat but with different settings
residual="--input_residual"

for char in A C L P; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in T F M G; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in B D H W; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

for char in R E N K; do
    CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
            --char $char \
            --margin $margin \
            --model_type $model_type \
            --batch_size $batch_size \
            --n_epochs $nepochs \
            --softmax_temperature $smt \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0
