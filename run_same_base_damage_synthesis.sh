#!/bin/bash

set -x
nr=5
ymd="2024-11-23"
# TODO: resize images to 224x224 here?
# TODO: how should $nr be set/how does it affect things?

# print start date
date

# cs="A B C D E F G H K L M N P Q R S T W"
# cs="A M P T W"
# cs="B C D E F G H K L N Q R S"
cs="C D E F G H K L N Q R S"
# cs="B"


for c in $cs; do
    python make_twin_dataset_from_splits.py \
        --char $c \
        --input_dir /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/char_${c}_uc \
        --output_dir /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/synthetic/${c}_twin_samebase_${nr}_${ymd} \
        --seed 42 \
        --binarize_image \
        --num_jobs 100 \
        --splits valid test \
        --num_repetitions ${nr} \
        --train_size 30000 \
        --valid_size 1500 \
        --test_size 1500 \
        --split_data_by_book_ratios 0.8 0.1 0.1
        # --debug
done

echo
echo
echo
echo
date

for c in $cs; do
    echo
    echo $c
    echo "Total:    $(find /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/synthetic/${c}_twin_samebase_${nr}_${ymd}/ -name "*.tif" | wc -l)"
    echo "Fracture: $(find /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/synthetic/${c}_twin_samebase_${nr}_${ymd}/ -name "*fracture*.tif" | wc -l)"
    echo "Bend:     $(find /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/synthetic/${c}_twin_samebase_${nr}_${ymd}/ -name "*bend*.tif" | wc -l)"
done

for c in $cs; do
    echo
    echo "---------------- $c ----------------"
    echo
    echo " ==== Fracture ===="
    echo find /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/synthetic/${c}_twin_samebase_${nr}_${ymd}/ -name "*fracture*.tif" | head -50 | xargs -I {} imgcat {}
    echo
    echo
    echo " ==== Bend ===="
    echo find /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/synthetic/${c}_twin_samebase_${nr}_${ymd}/ -name "*bend*.tif" | head -50 | xargs -I {} imgcat {}
done


####  rm -rf /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-*-out/synthetic/*_twin_samebase_5_2024-11-04
