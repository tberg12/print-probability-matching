#!/bin/bash
#SBATCH -c 20
#SBATCH --mem-per-cpu 1gb
#SBATCH -t 6:00:00
#SBATCH -p RM-shared
#set -x
#source ~/.bashrc
#conda activate /jet/home/nikolaiv/miniconda3/envs/py3

# parser = ArgumentParser('Creates a new twin dataset of train/valid/test images from random normal images. First, we sample an image, then we damage it, make two copies, and apply different inking perturbations to the copies. Loads perturbation settings from perturber_args.')
# parser.add_argument('--char', default='M')
# parser.add_argument('--binarize_image', action='store_true')
# parser.add_argument('--num_repetitions', default=20, type=int, help='Number of random perturbations to apply to a single image')
# parser.add_argument('--percent_apply_both_damage', default=0.0, help='Frequency to apply both bends and fractures to the same image.')  # was 0.03
# parser.add_argument('--splits_dir', default=f"/home/kishore/data/anomaly-detection/aligned-dataset/M/mixed_book_splits", help='Path to directory containing dataset splits')
# parser.add_argument('--src_dataframe_path', default=Path("/home/kishore/data/anomaly-detection/datasetv3"))
# parser.add_argument('--dest_dir', default="/home/kishore/data/anomaly-detection/aligned-dataset/M")
# parser.add_argument('--dest_dataset_name', default=f"M_aligned_perturbed20_twin_05_15")
# parser.add_argument('--num_jobs', default=4, type=int)
# parser.add_argument('--use_diff_base_images', action='store_true', help='Whether to generate twin damage pairs from aligning the skeletons of different base images, instead of using the same base image.')
# parser.add_argument('--merge_method', type=str, choices=['union', 'intersection'], default='union', help='Merge method for aligning twin images with different base images')
# args = parser.parse_args() if cl_args is None else parser.parse_args(shlex.split(cl_args))

# alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W']
#
#
nj=10  #100
#
#



# TODO: make this run on unaligned characters
# binarize here
# resize to 224x224 here?

if [ "$#" -ne 1 ]; then
    echo "Character required. Exiting."
    exit 1
fi

char=$1
collection_name=shakespeare2.0
ALIGNEDPREFIX=/trunk2/nvog/projects/git/damaged-type/detect_and_match/data/${collection_name}/${char}/images
splits_dir=data/synthetic/$collection_name/"$char"/splits

# NOTE:
#rm -rf $splits_dir

# for filter_filelist_by_ocular_logprob.py script
#top_n_pct=0.2

MATCHING=$PWD
# Restoration specific:
#ALIGNEDPREFIX=$SHARED/char_images3_aligned/restoration/char_images3
# Shakespeare:
#ALIGNEDPREFIX=$SHARED/char_images3_aligned/shakespeare/shakespeare
#collection_name=restoration
#collection_name=shakespeare
#splits_dir=data/$collection_name/"$char"/splits
mkdir -p $splits_dir/../normal
mkdir -p $splits_dir/../anomaly
mkdir -p $splits_dir/{train,valid,test}/{anomaly,normal}


pushd "$splits_dir"
find $ALIGNEDPREFIX/ -name "*.tif" | shuf > filelist_all_unsorted.txt
echo Found $(wc -l < filelist_all_unsorted.txt) total aligned $char files. Splitting into train/valid/test...
#echo Making Ocular emission prob sorted csv...
#rm filelist_all_logprob.csv
#while read -r line; do grep $(basename $line _aligned.tif) $SHARED/char_images3_csvs/$(basename $line _aligned.tif | cut -d'-' -f1)_color_uc.csv | cut -d',' -f2,5 | cut -d'/' -f2 >> filelist_all_logprob.csv; done < filelist_all_unsorted.txt
#$(basename /ocean/projects/hum160002p/shared/char_images3-A-out/A_aligned/jraworth_xxxxx_yyyyy_00height_englandslookingglasse-0064_page1rline12_char0_A_uc_aligned.tif _aligned.tif | cut -d'-' -f1)_color_uc.csv
#grep 'jraworth_xxxxx_yyyyy_00height_englandslookingglasse-0064_page1rline12_char0_A_uc' $SHARED/char_images3_csvs/jraworth_xxxxx_yyyyy_00height_englandslookingglasse_color_uc.csv | cut -d',' -f2,5 | cut -d'/' -f2
#cat filelist_all_logprob.csv | python -c "import sys; print('\n'.join([tup[0] for tup in sorted([(row.strip().split(',')[0].replace('.tif', '_aligned.tif'), float(row.strip().split(',')[1])) for row in sys.stdin], key=lambda x: -x[1])]))" > filelist_all.txt
#cat filelist_all_logprob.csv | python3 $MATCHING/filter_filelist_by_ocular_logprob.py $top_n_pct > filelist_all.txt
cp filelist_all_unsorted.txt filelist_all.txt
echo "*** filelist_all.txt created. ***"
wc -l filelist_all.txt
echo Splitting into 80/10/10 train/valid/test...
tot_lines=$(cat filelist_all.txt | wc -l)
train_amount=$(echo $tot_lines | python -c "import sys; print(round(float(sys.stdin.read().strip()) * 0.8))")
valid_amount=$(echo $tot_lines | python -c "import sys; print(round(float(sys.stdin.read().strip()) * 0.1))")
test_amount=$(($tot_lines - $train_amount - $valid_amount))
#test_amount=$(cat filelist_all.txt | wc -l | python -c "import sys, math; print(math.ceil(float(sys.stdin.read().strip()) * 0.1))")
head -$train_amount filelist_all.txt > filelist_train.txt
tail -n +$(($train_amount + 1)) filelist_all.txt | head -n $valid_amount > filelist_valid.txt
#tail -n +$(($train_amount + $test_amount + 2)) filelist_all.txt | head -n $test_amount > filelist_test.txt
tail -n $test_amount filelist_all.txt > filelist_test.txt
echo Copying the character images in these splits to local directories...
#prefix=$ALIGNEDPREFIX-"$char"-out/"$char"_aligned
#cp $(<$(cat filelist_train.txt | sed "s+^+$prefix+g")) train/normal/
#cp $(<$(cat filelist_valid.txt | sed "s+^+$prefix+g")) valid/normal/
#cp $(<$(cat filelist_test.txt | sed "s+^+$prefix+g")) test/normal/
for subset in train valid test; do 
    while read -r line; do cp $line $subset/normal/; cp $line ../normal/; done < filelist_"$subset".txt
done
#cp {train,valid,test}/normal/* ../normal/
popd
echo "*** Done compiling source character image lists. Ready for paired damage synthesis."
echo ""


nrep=10
# splits_dir="data/shakespeare/splits"
# src_dataframe_path="data/shakespeare"  # REMOVED, now we have dataset accept list of files and parse book name from this filename
# dest_dir="data/synthetic/shakespeare"
dest_dir="data/synthetic/$collection_name"
dest_dataset_name="$char"_twin_samebase_"$nrep"_"$(date +"%Y-%m-%d")"

[ ! -d "$splits_dir" ] && echo "$splits_dir does not exist." && exit 1
# [ ! -d "$src_dataframe_path" ] && echo "$src_dataframe_path does not exist." && exit 1

mkdir -p $dest_dir

date

python3 make_twin_dataset_from_splits.py \
        --char $char \
        --num_repetitions $nrep \
        --percent_apply_both_damage 0.0 \
        --splits_dir $splits_dir \
        --dest_dir $dest_dir \
        --dest_dataset_name $dest_dataset_name \
        --num_jobs $nj \
        --merge_method union
        # --use_diff_base_images \

        # --src_dataframe_path $src_dataframe_path \
date
