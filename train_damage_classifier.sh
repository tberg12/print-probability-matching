# REAL (cdt) original two class
# CSV_DIR=damage_classifier_data/dldt_ground_truth
# TRAIN=$CSV_DIR/train_pairs.csv
# VALID=$CSV_DIR/test_pairs.csv
# TEST=$CSV_DIR/test_pairs_01.csv

# REAL (cdt) manually filtered 3 class
CSV_DIR=damage_classifier_data
TRAIN=$CSV_DIR/3_class_data_train.csv
VALID=$CSV_DIR/3_class_data_test.csv
TEST=$CSV_DIR/3_class_data_test.csv

# SYNTHETIC
#CSV_DIR=data/synthetic_data
#TRAIN=$CSV_DIR/train_pairs.csv
#VALID=$CSV_DIR/valid_pairs.csv
# TEST=$CSV_DIR/test_pairs.csv
#TEST=damage_classifier_data/dldt_ground_truth/test.csv

# wandb=""
wandb="--wandb"

# ./A_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./B_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./C_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./D_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./E_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./F_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./G_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./H_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./K_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./L_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./M_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./N_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./P_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./Q_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./R_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./S_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./T_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
# ./W_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv


#
#  Create one synth dataset from separate character synth datasets
#
# echo > $TRAIN
# echo > $VALID
# echo > $TEST

# for c in A B C D E F G H K L M N P Q R S T W; do
#     echo $c
#     CTRAIN=$CSV_DIR/${c}_twin_samebase_noink_sz164_5_2025-03-16/train/pairs.csv
#     CVALID=$CSV_DIR/${c}_twin_samebase_noink_sz164_5_2025-03-16/valid/pairs.csv
#     CTEST=$CSV_DIR/${c}_twin_samebase_noink_sz164_5_2025-03-16/test/pairs.csv

#     # wc -l $CTRAIN

#     if [ $c == "A" ]; then
#         echo "copying header"
#         head -n 1 $CTRAIN > $TRAIN
#         head -n 1 $CVALID > $VALID
#         head -n 1 $CTEST > $TEST
#     fi

#     tail -n +2 $CTRAIN >> $TRAIN
#     tail -n +2 $CVALID >> $VALID
#     tail -n +2 $CTEST >> $TEST
# done

# resnet={18,34,50}, vit16, matcher models
# --size_transform fixed_height_resize_and_pad
#
# TODO: n_jobs
# CUDA_VISIBLE_DEVICES=0 python -m ipdb -c c damage_classifier.py $TRAIN $TEST $TEST damage_classifier_output/resnet34_frompretrained_lr0.001_outputs $wandb --use_resnet 34 --use_pretrained --batch_size 128 --lr 0.001 --patience 20 --size_transform matcher --use_amp
# CUDA_VISIBLE_DEVICES=0 python -m ipdb -c c damage_classifier.py $TRAIN $TEST $TEST damage_classifier_output/resnet50_frompretrained_lr0.001_outputs $wandb --use_resnet 50 --use_pretrained --batch_size 128 --lr 0.001 --patience 20 --size_transform matcher --use_amp
# CUDA_VISIBLE_DEVICES=0 python -m ipdb -c c damage_classifier.py $TRAIN $TEST $TEST damage_classifier_output/vit_b_16_frompretrained_lr0.001_outputs $wandb --use_vit 16 --use_pretrained --batch_size 512 --lr 0.001 --patience 20 --size_transform matcher --use_amp

# enc="resnet34"
enc="hf_hub:timm/vit_base_patch8_224.dino"
#enc="hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
if [ $enc == "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" ]; then
    bs=336
elif [ $enc == "hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k" ]; then
    bs=336
elif [ $enc == "hf_hub:timm/vit_base_patch8_224.dino" ]; then
    bs=168
elif [ $enc == "resnet34" ]; then
    bs=400
fi


enc_name="$(basename $enc)"
lr=0.0001
# CUDA_VISIBLE_DEVICES=6 python -m ipdb -c c damage_classifier.py $TRAIN $VALID $TEST damage_classifier_output/${enc_name}_frompretrained_lr${lr}_pair_outputs $wandb --encoder_type $enc --use_pretrained --batch_size $bs --lr $lr --patience 20 --size_transform matcher --use_amp --is_pair_data --jitter --apply_random_inking # --apply_position_jitter
# for real data
# CUDA_VISIBLE_DEVICES=1 python damage_classifier.py $TRAIN $VALID $TEST damage_classifier_output/${enc_name}_frompretrained_lr${lr}_pair_real_outputs $wandb --encoder_type $enc --use_pretrained --batch_size $bs --lr $lr --patience 20 --size_transform matcher --use_amp --is_pair_data --jitter --apply_random_inking --classes normal damaged # --apply_position_jitter

# for real data with manually annotated normals/bad extractions
# TODO: run on wandb
# add train set eval set?
# try with and without jitter
# try with and without pretraining
# try with synthetic damages replacing/augmenting the real ones
# try with and without binarization for real 3 class data
#
d=1
# for enc in "resnet34" "resnet50" "hf_hub:timm/vit_base_patch8_224.dino" "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"; do
# for enc in "resnet34" "resnet50"; do
#     if [ $enc == "hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" ]; then
#         bs=168
#     elif [ $enc == "hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k" ]; then
#         bs=168
#     elif [ $enc == "hf_hub:timm/vit_base_patch8_224.dino" ]; then
#         bs=84
#     elif [ $enc == "resnet34" ]; then
#         bs=400
#     elif [ $enc == "resnet50" ]; then
#         bs=200
#     fi
#     enc_name="$(basename $enc)"
#     tmux split-window -v "CUDA_VISIBLE_DEVICES=$d /graft2/code/nvog/git/matching/.venv/bin/python damage_classifier.py $TRAIN $VALID $TEST damage_classifier_output/${enc_name}_frompretrained_lr${lr}_real_3class_outputs $wandb --encoder_type $enc --use_pretrained --batch_size $bs --lr $lr --patience 20 --size_transform matcher --use_amp --classes normal damaged bad_extraction"
#     # tmux split-window -v "CUDA_VISIBLE_DEVICES=$d /graft2/code/nvog/git/matching/.venv/bin/python damage_classifier.py $TRAIN $VALID $TEST damage_classifier_output/${enc_name}_frompretrained_lr${lr}_real_3class_jitter_applyrandominking_outputs $wandb --encoder_type $enc --use_pretrained --batch_size $bs --lr $lr --patience 20 --size_transform matcher --use_amp --jitter --apply_random_inking --classes normal damaged bad_extraction"  # --apply_position_jitter 
#     # increment d
#     d=$((d+1))
#     if [ $d -gt 7 ]; then
#         d=0
#     fi
# done

CUDA_VISIBLE_DEVICES=7 python damage_classifier.py $TRAIN $VALID $TEST damage_classifier_output/${enc_name}_frompretrained_lr${lr}_real_outputs $wandb --encoder_type $enc --use_pretrained --batch_size $bs --lr $lr --patience 20 --size_transform matcher --use_amp --jitter --apply_random_inking --classes normal damaged bad_extraction # --apply_position_jitter

# for synthetic pair data
#CUDA_VISIBLE_DEVICES=1 python damage_classifier.py $TRAIN $VALID $TEST damage_classifier_output/${enc_name}_frompretrained_lr${lr}_pair_outputs $wandb --encoder_type $enc --use_pretrained --batch_size $bs --lr $lr --patience 20 --size_transform matcher --use_amp --is_pair_data --jitter --apply_random_inking --classes normal damaged # --apply_position_jitter
