

# char="A B C D E F G H K L M N P Q R S T W"
# char="A B C"
# char="A B C D E F G H K L M N P R S T W"
char="A B C D E F G H L M N P R S T W"
# char="B"
# patch16 dualencoder 168
# patch16 crossencoder 88
# patch8 dualencoder 84
# patch8 crossencoder 22
# NOTE: set below based off model_type and patch size
# train_batch_size=168
eval_batch_size=1  # NOTE: not used
#limit_train_pairs=15000
limit_train_pairs=30000
nepochs=60
evalfreq=1
margin=0.3
lr=0.0001
# model_type=DualEncoder
model_type=CrossEncoder
# encoder_type="vit_b_16"
# encoder_type="hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
# encoder_type='hf_hub:timm/vit_base_patch8_224.dino'
encoder_type='hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k'
# encoder_type='hf_hub:timm/convnext_base.fb_in22k_ft_in1k'
metric="gt_test_mix_neg"  # syn_valid
# npairs for DualEncoder, triplet_bce for CrossEncoder
# NOTE: set below based off model_type
# loss_type="triplet_bce"
# loss_type="npairs"
# attn_op="hw_sum"
# ncpb=3
# ncb=2


d=1
tnw=4
les=50
# for model_type in "CrossEncoder" "DualEncoder"; do
#for model_type in "CrossEncoder"; do
for model_type in "DualEncoder"; do
# for model_type in "CrossEncoder"; do
#     for encoder_type in "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" "hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k" "hf_hub:timm/vit_base_patch8_224.dino" "hf_hub:timm/convnext_base.fb_in22k_ft_in1k"; do
#     for encoder_type in "hf_hub:timm/convnext_base.fb_in22k_ft_in1k"; do
#     for encoder_type in "hf_hub:timm/vit_base_patch8_224.dino"; do
    for encoder_type in "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"; do
        if [ $model_type == "DualEncoder" ]; then
                if [ $encoder_type == "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" ]; then
                        # train_batch_size=84
                        train_batch_size=224
                        tnw=8
                elif [ $encoder_type == "hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k" ]; then
                        # train_batch_size=42
                        train_batch_size=56
                elif [ $encoder_type == "hf_hub:timm/vit_base_patch8_224.dino" ]; then
                        # train_batch_size=42
                        train_batch_size=56
                elif [ $encoder_type == "hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k" ]; then
                        # train_batch_size=42
                        train_batch_size=56
                elif [ $encoder_type == "hf_hub:timm/convnext_base.fb_in22k_ft_in1k" ]; then
                        # train_batch_size=42
                        train_batch_size=88
                #     elif [ $encoder_type == "hf_hub:timm/convnext_base.fb_in22k_ft_in1k" ]; then
                #         train_batch_size=84
                fi
                # loss_type="clip"
                loss_type="clip_extra"
        elif [ $model_type == "CrossEncoder" ]; then
                if [ $encoder_type == "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" ]; then
                        train_batch_size=88
                        tnw=8
                elif [ $encoder_type == "hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k" ]; then
                        train_batch_size=11
                elif [ $encoder_type == "hf_hub:timm/vit_base_patch8_224.dino" ]; then
                        train_batch_size=11
                fi
                loss_type="triplet_bce"
        fi

        date_str=$(date +'%Y-%m-%d_%H-%M-%S')

        # if d == 3 or 4, run this block
        # if [ $d -eq 3 ] || [ $d -eq 4 ]; then
                # residual="" # --conv_template_residual"
                # CUDA_VISIBLE_DEVICES=$d python -m ipdb -c c matcher.py \
        # CUDA_VISIBLE_DEVICES=$d .venv/bin/python -m ipdb -c c matcher.py \
        CUDA_VISIBLE_DEVICES=$d .venv/bin/python matcher.py \
                --char $char \
                --margin $margin \
                --synthetic_data_dir data/synthetic_data/normal_clf_data \
                --model_type $model_type \
                --encoder_type $encoder_type \
                --train_batch_size $train_batch_size \
                --eval_batch_size $eval_batch_size \
                --eval_interval $evalfreq \
                --optimizer Adam \
                --stopping_metric $metric \
                --output_dir output/matching_results_may25 \
                --log_dir output/matching_results_may25/logs \
                --n_epochs $nepochs \
                --eval_interval $evalfreq \
                --limit_dataset_size $limit_train_pairs \
                --limit_eval_dataset_size $les \
                --limit_synthetic_eval_set_size $les \
                --limit_gt_eval_background_set_size $les \
                --lr $lr \
                --augment \
                --add_global_inking \
                --amp \
                --train_num_workers $tnw \
                --eval_num_workers 2 \
                --loss_type $loss_type \
                --negative_mining random_half_normal \
                --use_pretrained \
                --scheduler linear_warmup_cosine_decay \
                --scheduler_warmup_steps 400 \
                --wandb &> output/matching_results_may25/train_${model_type}_d${d}_${date_str}.log
                # --debug   # specifically at looking for images in train_loader step by step
                # --train_num_workers 8 \
                # --eval_num_workers 2 \
                # TODO: implement the following
                # --add_random_background_inking 
                

                # -> WIP: implemented by default as s&p noise in make_jitter_transform, but for some reason it's rotating images 90 degrees wit extra ToTensor transform.
                # -> need to RandomApply this too so it doesnt always expect the noise on an input image
                # * z scale inputs?

                
                # --jitter_triplet \
                # --projection_net_out_dim 256 \
                # --projection_net_hidden_dim 256 \
                # --projection_net_type linear \

                # $residual \
                # --collapse_attn_operation $attn_op \
                # --num_cnn_blocks $ncb \
                # --num_conv_per_block $ncpb \
        # fi
        d=$((d+1))
        # if d == 8, then reset d to 0
        if [ $d -eq 8 ]; then
            d=0
        fi
    done
done
