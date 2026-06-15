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
# setting=big_letters
#query_setting=lists/manual_REDOfnamesall_charimages3_paths.txt
# query_setting=lists/auto_top5pctbychartwotreatisesC.txt
# candidate_setting=char_images3_REDO_valid_2025-05-27_damagepredictions_top20pct
# query_setting=lists/auto_top0.1pctbigletters.txt
# query_setting=lists/auto_top1pctbigletters_anon.txt
# query_setting=lists/auto_top1pctbigletters.txt

#query_setting=lists/manualfortysermons_paths.txt
query_setting=lists/manualvotes_paths.txt
#candidate_setting=fortysermonsREDO

# TODO: reset to this
#query_setting=lists/manuallocke_paths.txt

#candidate_setting=char_images3_REDO_valid_2025-05-27_damagepredictions_top20pct
#candidate_setting=englandsimprovement
#candidate_setting=goodcheaphusbandry
#candidate_setting=char_images3
#candidate_setting=macock
#candidate_setting=amaxwell
#candidate_setting=bookofjob
#candidate_setting=anekdota
#candidate_setting=milbourn
#candidate_setting=ssmith
#candidate_setting=bentley
#candidate_setting=1680sAnd1690s
#candidate_setting=criticalenquiries
#candidate_setting=dejure
#candidate_setting=religionandreason
#candidate_setting=braddyll
candidate_setting=braddylleveringham


# query_setting=lists/dldt_char_paths.txt
# candidate_setting=anon_redo_53books_chars_valid
# candidate_setting=anon_redo_and_restoration_chars
# setting=gt_test_lockespinoza_mix_neg-N

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
--k 50 \
--faiss_image_paths_file lists/faiss_image_paths_${candidate_setting}.txt \
--faiss_index_path output/faiss_index_${candidate_setting}.index \
--faiss_sim_mat_path output/faiss_index_${candidate_setting}.npy \
--faiss_query_set_file $query_setting

#
# NOTE: currently the faiss sim mat path code is commented out in matcher.py so it doesn't compute/save it
#

# --k 10 \

