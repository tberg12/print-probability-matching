#
# SBATCH CRAP
#
#set -x
#source ~/.bashrc
#conda activate /jet/home/nikolaiv/miniconda3/envs/py3

if [ "$#" -ne 4 ]; then
    echo "[char] [folio_number] [part] [damage_thresh] required. Exiting."
    exit 1
fi

char=$1
nfolio=$2
part=$3
damage_thresh=$4
#damage_thresh=0.8


# gpu device id
d=0
batch_size=4096  # currently ignored in matcher, as the entire candidate set is considered at a time
split_amount=3
ckptdir=/trunk/nvog/matching_results
#ckptdir=/ocean/projects/hum160002p/nikolaiv/projects/git/anomaly-detection/matching/checkpoints
ckptname=$(printf "%s\n" $ckptdir/Char${char}*Batchsize512*2021* | head -1)  # get most recent ckpt for character
epoch=best.pt
ckpt=$ckptname/$epoch
echo $ckpt

exp_dir=experiments/shakespeare${nfolio}foliopart${part}/$char
mkdir $exp_dir
filelist=$exp_dir/filelist.txt
aligned_data=/projects/nvog/char_images3_aligned/shakespeare_2021-09-28/shakespeare-${char}-out/${char}_aligned
find $aligned_data -name "*aligned.tif" | python3 filter_shakespeare_books_by_folio_number.py ${nfolio}p${part} > $filelist

# create two filelists, shakespeare vs. other, to match against
shakespeare_filelist=$exp_dir/shakespeare_filelist.txt
other_filelist=$exp_dir/other_filelist.txt
cat $filelist | grep shakespeare${nfolio} > $shakespeare_filelist
cat $filelist | grep -v shakespeare${nfolio} > $other_filelist

# filelist=data/shakespeare${pct}/filelist_allbooks.txt
#filelist=data/shakespeare${pct}/$char/filelist${nfolio}.txt

# concatenate all book filelists and run damage detection
# rm $filelist_allbooks
#rm $filelist
#cat $book_filelists | python3 filter_shakespeare_books_by_folio_number.py $nfolio > $filelist
# for f in $filelists; do
#     cat $f >> $filelist_allbooks
# done
echo
echo Folio $nfolio part $part file list:
echo "Query/Interesting:"
echo $shakespeare_filelist
wc -l $shakespeare_filelist
echo "Candidates/Others:"
echo $other_filelist
wc -l $other_filelist
echo

#echo NOTE: skipping damage detection for now!
# load already trained damage classifier from file
det_dir=$exp_dir/detector_output
filelist_damages=$det_dir/$(basename $filelist .txt)_damagethresh${damage_thresh}.txt
echo $det_dir
echo $filelist_damages
CUDA_VISIBLE_DEVICES=$d python3 damage_classifier.py \
    $det_dir/train_data.csv \
    $det_dir/valid_data.csv \
    $det_dir/test_data.csv \
    $det_dir \
    --evaluate $filelist \
    --loss_type object_ce \
    --output_pooling none \
    --batch_size 512 \
    --load_model $det_dir/best.pt
# filter files above damage thresh predicted by classifier
cat $det_dir/$(basename $filelist .txt)_damagepredictions.csv | 
    python -c "import sys; print('\n'.join([line.strip() for line in sys.stdin if float(line.strip().split(',')[1]) > $damage_thresh]))" | 
    cut -d, -f1 > $filelist_damages
echo
echo $filelist_damages
wc -l $filelist_damages
echo

echo ****
echo TODO: do within-book matching to add more damaged characters to the filelist that may have been missed by the damage detector
echo ****

output_dest=$exp_dir/matching_output/damagethresh${damage_thresh}
mkdir -p $output_dest

interesting_filelist=$shakespeare_filelist
candidates_filelist=$other_filelist

echo
echo  **** NOTE: currently using combined filelist, instead of separate query/candidate lists
echo

date

#exit 1

CUDA_VISIBLE_DEVICES=$d python3 matcher.py \
        --char $char \
        --model_type Attention \
        --batch_size $batch_size \
        --megabatch_size 4 \
        --negative_mining UniformSampling \
        --n_epochs 80 \
        --softmax_temperature 1.0 \
        --eval_interval 1 \
        --optimizer Adam \
        --lr 0.0001 \
        --conv_template_residual \
        --collapse_attn_operation hw_sum \
        --load_model "$ckpt" \
        --evaluate_ckpt \
        --evaluate_ckpt_dest "$output_dest" \
        --discover_matches "$filelist_damages" "$filelist_damages" \
        --discover_matches_split_amount $split_amount \
        --exclude_same_book_matches

        #--discover_matches "$interesting_filelist" "$candidates_filelist" \
date
