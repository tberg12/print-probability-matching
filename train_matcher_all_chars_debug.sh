
d=0
max_gpu=3
# A B C D E F G H K L M N P R T W
char=A
batch_size=64
nepochs=121
evalfreq=10
margin=0.3
lr=0.0001
attn_op="hw_sum"
model_type=Attention # Attention
ncb=2
ncpb=3
smt=1.0
metric="syn_valid"
lds=1000 # TODO

# TODO
for char in D; do
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
            --limit_dataset_size $lds \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric  \
            --notes testinking
            #--wandb &
            #--jitter_triplet \
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

exit

residual="--conv_template_residual"
for char in D F G M; do
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
            --limit_dataset_size $lds \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric  \
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


residual="--input_residual"
for char in D F G M; do
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
            --limit_dataset_size $lds \
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

for char in D F G M; do
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
            --limit_dataset_size $lds \
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

for char in D F G M; do
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
            --limit_dataset_size $lds \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric \
            --jitter_triplet \
            --wandb &
            #--l2_baseline
    d=$((d+1))
    #if [ "$d" -gt "$max_gpu" ]; then
    #    d=0
    #fi
done

wait
d=0

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
            --limit_dataset_size $lds \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric  \
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
            --limit_dataset_size $lds \
            $residual \
            --collapse_attn_operation $attn_op \
            --num_cnn_blocks $ncb \
            --num_conv_per_block $ncpb \
            --stopping_metric $metric  \
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
# TODO TODO TODO
exit

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
            --limit_dataset_size $lds \
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

attn_op="filter_sum"
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
            --limit_dataset_size $lds \
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

attn_op="hw_sum"
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
            --limit_dataset_size $lds \
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

lds=10000
attn_op="filter_sum"
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
            --limit_dataset_size $lds \
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
