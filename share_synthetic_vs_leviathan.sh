#!/bin/bash

# set -x
nr=5
ymd="2024-11-06"
# TODO: resize images to 224x224 here?
# TODO: how should $nr be set/how does it affect things?

# print start date
date

# only chars that also have real leviathan test set data
cs="A B C D E F G H K L M N P R T W"

rm -f synthetic_paths.txt
rm -f synthetic_pairs.csv
rm -f leviathan_pairs.csv
rm -f leviathan_preprocessed_pairs.csv

# sample 5 bend pairs and 5 fracture pairs from each character (10 total from each)
for c in $cs; do
    find /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/synthetic/${c}_twin_samebase_${nr}_${ymd}/ -name "*fracture*.tif" | sort | head -10
    find /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/synthetic/${c}_twin_samebase_${nr}_${ymd}/ -name "*bend*.tif" | sort | head -10
done > synthetic_paths.txt

# convert synthetic paths to a 2-column csv. Consecutive rows are pairs in the same row.
awk 'NR%2{printf "%s,",$0;next}1' synthetic_paths.txt > synthetic_pairs.csv

# sample real paths from leviathan
for c in $cs; do
    # this is a csv containing filenames so only take the first two columns of each row
    shuf -n 10 /home/nvog/projects/git/anomaly-detection/matching/leviathan_matching_test_set/${c}/test/matches_unaligned.csv | cut -d, -f1,2
done > leviathan_pairs.csv

# delete lines from leviathan_pairs.csv that dont have a comma
sed -i '/,/!d' leviathan_pairs.csv


# for each file in leviathan_pairs.csv, process the image with `python preprocess_image.py` and save the output to a new directory with a similar pairs getting the same id
output_dir="/home/nvog/projects/git/damaged-type/detect_and_match/leviathan_preprocessed"

mkdir -p $output_dir
while IFS=, read -r f1 f2; do
    # parse char from f1
    c=$(echo $f1 | cut -d_ -f8)
    echo $c
    full_path_prefix="/trunk2/nvog/char_images3_aligned/leviathanornaments_2022-05-15/leviathanornaments-${c}-0.001-out/char_${c}_uc"
    # find the input files full path at full_path_prefix
    python preprocess_image.py --input $full_path_prefix/$f1 --output $output_dir/$(basename $f1)
    python preprocess_image.py --input $full_path_prefix/$f2 --output $output_dir/$(basename $f2)
    # echo this pair to a csv
    echo "$output_dir/$(basename $f1),$output_dir/$(basename $f2)" >> leviathan_preprocessed_pairs.csv
done < leviathan_pairs.csv

cat leviathan_preprocessed_pairs.csv synthetic_pairs.csv | shuf > all_pairs.csv
# cat leviathan_preprocessed_pairs.csv