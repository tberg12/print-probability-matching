#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 03:00:00
#SBATCH --gpus=1


# uncomment for bridges2:
# module load cuda/10.2.0
# eval "$(/ocean/projects/hum160002p/nikolaiv/miniconda3/bin/conda shell.bash hook)"
# conda activate py3
# which python3


if [ "$#" -ne 2 ]; then
    echo "[filelist] [out_dir] required. Exiting."
    exit 1
fi

d=4
filelist=$1
out_dir=$2
# for bridges2:
# batch_size=512
batch_size=6144

# for bridges2:
# here=/ocean/projects/hum160002p/nikolaiv/git/damaged-type/detect_and_match
here=/graft2/code/nvog/git/matching
# run_name="resnet34_frompretrained_lr0.001"
run_name="resnet34_frompretrained_lr0.0001_real_3class"
# for bridges2:
# model_dir=$here/${run_name}_outputs
model_dir=$here/damage_classifier_output/${run_name}_outputs
model_name=best.pt


mkdir -p $out_dir

echo Predicting damage on $filelist
echo Putting results in "$out_dir"

# NOTE: I'm not transferring the training/validation set from savernake for time/space purposes,
# so we'll just load the old train/val/test dataset since we need to pass it as an arg but it
# has no effect anyways during evaluation time on a file list
# data_dir=$here/old_damage_detector/detector_output
# TRAIN=$data_dir/real_train_data_aligned.csv
# VAL=$data_dir/real_test_data_aligned.csv
# TEST=$data_dir/real_test_data_aligned.csv

# data_dir=$here/damage_classifier_data/dldt_ground_truth
# TRAIN=$data_dir/train.csv
# VAL=$data_dir/test.csv
# TEST=$data_dir/test.csv
data_dir=$here/damage_classifier_data
TRAIN=$data_dir/3_class_data_train.csv
VAL=$data_dir/3_class_data_test.csv
TEST=$data_dir/3_class_data_test.csv

# don't use amp on bridges2?
# args="--encoder_type resnet34 --size_transform matcher"
# args="--encoder_type resnet34 --size_transform matcher --use_amp"
args="--encoder_type resnet34 --size_transform matcher --use_amp --classes normal damaged bad_extraction"

script_dir=$here
script=$script_dir/damage_classifier.py

date
# old bridges2:
#CUDA_VISIBLE_DEVICES=$d /jet/home/nikolaiv/miniconda3/envs/py3/bin/python3.8 $script \
CUDA_VISIBLE_DEVICES=$d python3 $script \
    $TRAIN \
    $VAL \
    $TEST \
    $out_dir \
    --evaluate $filelist \
    --batch_size $batch_size \
    --load_model $model_dir/$model_name \
    --num_workers 12 \
    $(echo $args)
date
