
d=''
max_gpu=3
# A B C D E F G H K L M N P R T W
char=A
batch_size=64
nepochs=101
evalfreq=10
margin=0.3
lr=0.0001
attn_op="hw_sum"
model_type=Attention # Attention


#CUDA_VISIBLE_DEVICES=$d python3 -m pdb matcher.py \
for char in A B C D E F G H K L M N P R T W; do
    for bl in L2; do  # L2, random
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
                --baseline_type $bl &
    done
done

