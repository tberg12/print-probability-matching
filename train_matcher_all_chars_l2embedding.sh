
d=0
max_gpu=3
# A B C D E F G H K L M N P R T W
char=A
batch_size=64
nepochs=101
evalfreq=10
margin=0.3
lr=0.0001
attn_op="hw_sum"
model_type=L2Embedding  # Attention


for char in A B C D; do
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

for char in A B C D; do
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
            --input_residual \
            --collapse_attn_operation $attn_op \
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


#CUDA_VISIBLE_DEVICES=$d python3 -m pdb matcher.py \
for char in A C L P; do
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

for char in T F M G; do
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
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
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
            --softmax_temperature 1.0 \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            --conv_template_residual \
            --collapse_attn_operation $attn_op \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
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
            --softmax_temperature 1.0 \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            --conv_template_residual \
            --collapse_attn_operation $attn_op \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
done

wait
d=0

for char in A C L P; do
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
            --input_residual \
            --collapse_attn_operation $attn_op \
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

for char in T F M G; do
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
            --input_residual \
            --collapse_attn_operation $attn_op \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
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
            --softmax_temperature 1.0 \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            --input_residual \
            --collapse_attn_operation $attn_op \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
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
            --softmax_temperature 1.0 \
            --eval_interval $evalfreq \
            --optimizer Adam \
            --lr $lr \
            --input_residual \
            --collapse_attn_operation $attn_op \
            --wandb &
            #--jitter_triplet \
            #--conv_template_residual \
            #--l2_baseline
    d=$((d+1))
done
