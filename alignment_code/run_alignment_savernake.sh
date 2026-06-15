#!/bin/bash
# run_alignment_savernake.sh
# NOTE: without shear and with init_avg

# CHARS=(A B C D E F G H K L M N P Q R S T W)
SHARED_DIR=/trunk2/nvog/shared
ALIGNMENT_DIR=/home/nvog/projects/git/damaged-type/detect_and_match/alignment_code
ALIGNIN=/trunk2/nvog/shared/char_images3_alignment_input/shakespeare
ALIGNOUT=/trunk2/nvog/shared/char_images3_aligned/shakespeare_2024-10-26
save_prefix=$ALIGNOUT/char
lr=0.1

# for c in ${CHARS[@]}; do
#     # alignment ckpt
#     ckpt="placeholder"
#     echo "Status: Prepping alignment for ${c}"
#     bash $ALIGNMENT_DIR/prep_untar_alignstretch_slurm_train_from_ja.sh $ALIGNIN/tars/ $ALIGNIN/csvs/ $c $save_prefix $ckpt $lr
# done

c=S
lr=0.1
epochs=50
lbd=0.1   # was 0.05  # l2 regularization on templates
ink_var=0.2
y_pix=64
batch_size=32
data="${ALIGNOUT}/char-${c}-${lr}-out"

# python -m pdb -c c pnp_alignment/alignment/aligner_stretch.py \

CUDA_VISIBLE_DEVICES=1 python pnp_alignment/alignment/aligner_stretch.py \
    -ink_var $ink_var \
    -data "$data" \
    -aligncsv "${c}_align.csv" \
    -init_avg \
    -nc 1 \
    -y_pix "$y_pix" \
    -gpu 0 \
    -epochs "$epochs" \
    -optim adam \
    -adj_learning_rate "$lr" \
    -visualize \
    -batch_size "$batch_size" \
    -output "${c}_template" \
    -rec "${c}_aligned" \
    -save_dir "${c}_model" \
    -lbd "$lbd" \
    -no_shear \
    -freeze_template_params
