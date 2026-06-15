# char="A B C D E F G H K L M N P Q R S T W"
# char="A B C"
# char="A B C D E F G H K L M N P R S T W"
# NOTE: just do whole alphabet of manual queries
char="A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
# char="N"
# patch16 dualencoder 168
# patch16 crossencoder 88
# patch8 dualencoder 84
# patch8 crossencoder 22
# NOTE: set below based off model_type and patch size
# train_batch_size=168
eval_batch_size=1  # NOTE: not used
limit_train_pairs=1000
evalfreq=1
nepochs=60
margin=0.3
lr=0.0001
metric="gt_test_mix_neg"  # syn_valid

limit_eval_size=100

# ckpt="output/matching_results_nov24/CharABCDEFGHKLMNPRSTW_ModelDualEncoder_EncoderVit_base_patch8_224.dino_15000TrainPairs_LimitEval50_GlobalInkTrue_Pretrained_Temp1.0_Augment_NPairs_Bsz84_AdamLR0.0001_SchedulerWU500-2024-12-14_13:15:30/best.pt"
ckpt="output/matching_results_may25/CharABCDEFGHLMNPRSTW_ModelDualEncoder_EncoderVit_base_patch16_224.augreg2_in21k_ft_in1k_30000TrainPairs_LimitEval50_GlobalInkTrue_Pretrained_Temp1.0_Augment_CLIPExtra_Bsz224_AdamLR0.0001_SchedulerWU400-2025-05-27_00:36:08/best.pt"
model_type="DualEncoder"
# encoder_type="hf_hub:timm/vit_base_patch8_224.dino"
encoder_type="hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"


d=2
tnw=2
les=50

# for query_set in "manualcriticalenquiries_fnames.csv"; do
for query_set in "manuallockespinozanature.csv"; do
        for candidate_set in "lockespinozanature_candidates.txt"; do
                echo "Discovering matches between $query_set and $candidate_set"
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
                        --k 10 \
                        --load_model $ckpt \
                        --discover_matches lists/${query_set} lists/${candidate_set} \
                        --evaluate_ckpt_dest output/evaluate_ckpt_may25
                        # --wandb &> output/matching_results_may25/train_${model_type}_d${d}_${date_str}.log
        done
done