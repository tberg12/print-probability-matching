

d=6
# char="A B C D E F G H K L M N P Q R S T W"
# char="A B C"
char="A B C D E F G H K L M N P R S T W"
# char="N"
# patch16 dualencoder 168
# patch16 crossencoder 88
# patch8 dualencoder 84
# patch8 crossencoder 22
# NOTE: set below based off model_type and patch size
# train_batch_size=168
eval_batch_size=1  # NOTE: not used
limit_train_pairs=15000
evalfreq=1
margin=0.3
metric="gt_test_mix_neg"  # syn_valid

limit_eval_size=100

ckpt="output/matching_results_nov24/CharABCDEFGHKLMNPRSTW_ModelCrossEncoder_EncoderVit_base_patch16_224.augreg2_in21k_ft_in1k_15000TrainPairs_LimitEval50_GlobalInkTrue_Pretrained_Temp1.0_Augment_TripletBCE-random_Bsz88_AdamLR0.0001_SchedulerWU500-2024-12-14_13:15:30/best.pt"
model_type="CrossEncoder"
encoder_type="hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"

#
#
# TODO: add random background inking (stop it rotating images)
# 
#
#
# for model_type in "CrossEncoder" "DualEncoder"; do
#     for encoder_type in "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" "hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k" "hf_hub:timm/vit_base_patch8_224.dino"; do
if [ $model_type == "DualEncoder" ]; then
        if [ $encoder_type == "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" ]; then
                train_batch_size=168
        elif [ $encoder_type == "hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k" ]; then
                train_batch_size=84
        elif [ $encoder_type == "hf_hub:timm/vit_base_patch8_224.dino" ]; then
                train_batch_size=84
        #     elif [ $encoder_type == "hf_hub:timm/convnext_base.fb_in22k_ft_in1k" ]; then
        #         train_batch_size=84
        fi
        loss_type="npairs"
elif [ $model_type == "CrossEncoder" ]; then
        if [ $encoder_type == "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" ]; then
                train_batch_size=88
        elif [ $encoder_type == "hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k" ]; then
                train_batch_size=22
        elif [ $encoder_type == "hf_hub:timm/vit_base_patch8_224.dino" ]; then
                train_batch_size=22
        fi
        loss_type="triplet_bce"
fi

date_str=$(date +'%Y-%m-%d_%H-%M-%S')


d=2
# for query_set in "manualcriticalenquiries_fnames.csv"; do
for query_set in "manuallocke_fnames.csv"; do
        # for candidate_set in "everingham_english.txt" "braddyll.txt" "forty_sermons.txt" "locke_letter.txt" "locke_two_treatises.txt" "braddyll_critical.txt" "braddyll_kingjames.txt" "braddyll_plutarch.txt"; do
        # for candidate_set in "everingham_anekdota.txt" "everingham_ogygia.txt" "everingham_examen.txt"; do
        # for candidate_set in "braddyll_religionandreason.txt"; do
        # for candidate_set in "everingham_ogygia.txt" "everingham_examen.txt" "spinoza_theological.txt" "braddyll_twobooks.txt"; do
        for candidate_set in "braddyll_twobooks.txt" "braddyll_voyageofitaly.txt" "braddyll.txt" "everingham_english.txt"; do
                echo "Discovering matches between $query_set and $candidate_set"
                # CUDA_VISIBLE_DEVICES=$d python -m ipdb -c c matcher.py \
                CUDA_VISIBLE_DEVICES=$d python matcher.py \
                        --char $char \
                        --margin $margin \
                        --model_type $model_type \
                        --encoder_type $encoder_type \
                        --train_batch_size $train_batch_size \
                        --eval_batch_size $eval_batch_size \
                        --eval_interval $evalfreq \
                        --optimizer Adam \
                        --stopping_metric $metric \
                        --output_dir output/matching_results_nov24 \
                        --log_dir output/matching_eval_nov24/logs \
                        --eval_interval $evalfreq \
                        --limit_dataset_size $limit_train_pairs \
                        --limit_eval_dataset_size $limit_eval_size \
                        --limit_synthetic_eval_set_size $limit_eval_size \
                        --limit_gt_eval_background_set_size $limit_eval_size \
                        --amp \
                        --train_num_workers 8 \
                        --eval_num_workers 2 \
                        --loss_type $loss_type \
                        --use_pretrained \
                        --k 10 \
                        --load_model $ckpt \
                        --discover_matches lists/${query_set} lists/${candidate_set} \
                        --evaluate_ckpt_dest output/evaluate_ckpt \
                        --query_parts C
                        # --discover_matches lists/spinoza_theological.txt lists/locke_letter.txt \
                        # --candidate_parts A
                        # --exclude_same_book_matches
                        # --exclude_book_name_from_candidates \
                        # --restrict_query_to fortysermons1685 \

                        # --debug   # specifically at looking for images in train_loader step by step
                        # --discover_matches_split_amount $eval_batch_size
                        # --evaluate_ckpt_dest evaluate_ckpt \
                        # --evaluate_ckpt #&> output/matching_eval_nov24/eval_${ckpt}_d${d}_${date_str}.log
                        # --wandb
                        # --discover_matches lists/manualspinoza_fnames.csv lists/locke_letter.txt \
                        # --discover_matches lists/manualspinoza_fnames.csv lists/everingham.txt \
                        # --discover_matches lists/manualspinoza_fnames.csv lists/everingham_english.txt \
                        # --discover_matches lists/manualspinoza_fnames.csv lists/braddyll.txt \
                        # --discover_matches lists/manualspinoza_fnames.csv lists/forty_sermons.txt \
        done
done