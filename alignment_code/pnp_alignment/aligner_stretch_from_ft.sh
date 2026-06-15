#!/bin/bash

#python alignment/model_vae.py -ink_var 0.2 -data agg_color/char_M_uc/ -output templates_vae -rec rec_vae -init_rand -nc 1 -y_pix 56 -gpu 1 -epochs 200 -optim adam -learning_rate 0.001 -visualize -batch_size 24 -lbd 0.05 -encode_method shift_template -adj_first -nosave

set -x
which python
which python3
batch_size=64
epochs=3000
lr=$5
# train from scratch
# NOTE: no_binarize flag
# NOTE -init_rand is default but gets overridden automatically in code by -train_from checkpoint saved template
python alignment/aligner_stretch.py -ink_var 0.2 -data $1 -aligncsv $2"_align.csv" -init_rand -nc 1 -y_pix 64 -gpu $3 -epochs $epochs -optim adam -adj_learning_rate "$lr" -visualize -batch_size $batch_size -output $2"_template" -rec $2"_aligned" -save_dir $2"_model" -lbd 0.05 -train_from $4 #-freeze_template_params
#-no_binarize
# TODO: load from checkpoint using -init_dir instead of -init_rand -- doesnt matter 
#python alignment/aligner.py -ink_var 0.2 -data $1 -aligncsv $2"_align.csv" -init_dir -nc 1 -y_pix 64 -gpu $3 -epochs $epochs -optim adam -learning_rate 0.001 -visualize -batch_size $batch_size -output $2"_template" -rec $2"_aligned" -save_dir $2"_model" -lbd 0.05
