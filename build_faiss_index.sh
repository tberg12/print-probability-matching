#!/bin/bash

# Prerequisites:
# 1. create input filelist
# e.g., find $PWD/big_letters -name "*.jpg" > lists/faiss_image_paths_big_letters.txt

# encoder_type="hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
# encoder_type="hf_hub:timm/vit_base_patch8_224.dino"
encoder_type="hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
# ckpt="output/matching_results_nov24/CharABCDEFGHKLMNPRSTW_ModelDualEncoder_EncoderVit_base_patch8_224.dino_15000TrainPairs_LimitEval50_GlobalInkTrue_Pretrained_Temp1.0_Augment_NPairs_Bsz84_AdamLR0.0001_SchedulerWU500-2024-12-14_13:15:30/best.pt"
# ckpt="output/matching_results_may25/CharABCDEFGHLMNPRSTW_ModelDualEncoder_EncoderVit_base_patch16_224.augreg2_in21k_ft_in1k_15000TrainPairs_LimitEval50_GlobalInkTrue_Pretrained_Temp1.0_Augment_CLIPExtra_Bsz224_AdamLR0.0001_SchedulerWU400-2025-05-20_16:10:46/best.pt"
ckpt="output/matching_results_may25/CharABCDEFGHLMNPRSTW_ModelDualEncoder_EncoderVit_base_patch16_224.augreg2_in21k_ft_in1k_30000TrainPairs_LimitEval50_GlobalInkTrue_Pretrained_Temp1.0_Augment_CLIPExtra_Bsz224_AdamLR0.0001_SchedulerWU400-2025-05-27_00:36:08/best.pt"
d=0

# setting=gt_test_mix_neg-A  # big_letters, gt_test_mix_neg, gt_test_lockespinoza_mix_neg
# setting=gt_test_lockespinoza_mix_neg-N
# setting=big_letters
# setting=fortysermons
#setting=fortysermonsREDO

# setting=anon_redo_53books_chars_valid
# setting=anon_redo_and_restoration_chars
#setting=char_images3_REDO_valid_2025-05-27_damagepredictions_top20pct
#setting=englandsimprovement
#setting=goodcheaphusbandry
#setting=char_images3
#setting=macock
#setting=amaxwell
#setting=bookofjob
#setting=anekdota
#setting=milbourn
#setting=ssmith
#setting=bentley
#setting=1680sAnd1690s
#setting=criticalenquiries
#setting=dejure
#setting=religionandreason
#setting=braddyll
setting=braddylleveringham

# this gets about 30k damaged matches and combined hard/easy/mix neg background
# find $PWD/data/leviathan_matching_test_set_preprocessed/A/test/ -name "*.tif" > lists/faiss_image_paths_${setting}.txt
# 1008:
# find $PWD/data/lockespinoza_matching_test_set/N/test/ -name "*.tif" > lists/faiss_image_paths_${setting}.txt

CUDA_VISIBLE_DEVICES=$d python matcher.py \
--model_type DualEncoder \
--encoder_type $encoder_type \
--use_pretrained \
--load_model $ckpt \
--output_dir output/matching_results_may25 \
--log_dir output/matching_eval_may25/logs \
--amp \
--eval_batch_size 4096 \
--eval_num_workers 12 \
--faiss_build_index \
--faiss_image_paths_file lists/faiss_image_paths_${setting}.txt \
--faiss_index_path output/faiss_index_${setting}.index \
--faiss_sim_mat_path output/faiss_index_${setting}.npy \

# can query this index with the query_faiss_index.sh script

#
# NOTE: currently the faiss sim mat path code is commented out in matcher.py so it doesn't compute/save it
#

# --k 10 \

