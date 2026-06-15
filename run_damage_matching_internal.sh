#
# SBATCH CRAP
#
#set -x
#source ~/.bashrc
#conda activate /jet/home/nikolaiv/miniconda3/envs/py3

if [ "$#" -ne 5 ]; then
    echo "[char] [shakespeare_part] [exp] [device] [damage_thresh] required. Exiting."
    exit 1
fi

char=$1
shakespeare_part=$2
exp=$3  # macock_matching, rroberts_matching, leviathan_matching
d=$4
#nfolio=$2
#part=$3
damage_thresh=$5
restrict_query_to=behemoth1679    # leviathanornaments  # shakespeare4folio1685
filelist_queries=experiments/behemoth/${char}/behemoth_a-h_queries_2022-07-05_new_aligned.txt  #leviathanornaments/${char}/leviathanornaments_queries_2022-05-27_aligned.txt
#damage_thresh=0.0


# gpu device id
#d=2
batch_size=4096  # currently ignored in matcher, as the entire candidate set is considered at a time
split_amount=3
ckptdir=/trunk/nvog/matching_results
#ckptdir=/ocean/projects/hum160002p/nikolaiv/projects/git/anomaly-detection/matching/checkpoints
ckptname=$(printf "%s\n" $ckptdir/Char${char}*Batchsize512*2021* | tail -1)  # get most recent ckpt for character
epoch=best.pt
ckpt=$ckptname/$epoch
echo $ckpt

exp_dir=experiments/$exp/$char
filelist=$exp_dir/matching_output/filelist_all.txt

# concatenate all book filelists and run damage detection
# rm $filelist_allbooks
#rm $filelist
#cat $book_filelists | python3 filter_shakespeare_books_by_folio_number.py $nfolio > $filelist
# for f in $filelists; do
#     cat $f >> $filelist_allbooks
# done

#echo NOTE: skipping damage detection for now!
# load already trained damage classifier from file
det_dir=$exp_dir/detector_output
#model_dir=~/projects/git/anomaly-detection/matching/experiments/shakespeare4foliopart1/$char/detector_output
model_dir=~/projects/git/anomaly-detection/matching/experiments/shakespeare4foliopart1/aggregated_all_chars/detector_output
#model_name=best.pt
model_name=ce-0.3-embres--_best_epoch.pt
filelist_damages=$det_dir/$(basename $filelist .txt)_damagethresh${damage_thresh}.txt
echo $det_dir
echo $filelist_damages
CUDA_VISIBLE_DEVICES=$d python3 ~/projects/git/anomaly-detection/matching/damage_classifier.py \
    $model_dir/real_train_data_aligned.csv \
    $model_dir/valid_data.csv \
    $model_dir/real_test_data_aligned.csv \
    $det_dir \
    --evaluate $filelist \
    --loss_type ce \
    --output_pooling none \
    --batch_size 512 \
    --load_model $model_dir/$model_name \
    --use_template_residual
    #$det_dir/train_data.csv \
    #$det_dir/valid_data.csv \
    #$det_dir/test_data.csv \

# filter files above damage thresh predicted by classifier
# BY THRESHOLD FLOAT VALUE 
cat $det_dir/$(basename $filelist .txt)_damagepredictions.csv | 
    python -c "import sys; print('\n'.join(sorted([line.strip() for i, line in enumerate(sys.stdin) if float(line.strip().split(',')[1]) > $damage_thresh], key=lambda x: -float(x.strip().split(',')[1]))))" | 
    cut -d, -f1 > $filelist_damages
# BY TOP AMOUNT
#cat $det_dir/$(basename $filelist .txt)_damagepredictions.csv | 
#    python -c "import sys; print('\n'.join(sorted([line.strip() for i, line in enumerate(sys.stdin)], key=lambda x: -float(x.strip().split(',')[1]))[:$damage_thresh]))" | 
#    cut -d, -f1 > $filelist_damages

#echo NOTE: skipping damage detection for just internal matching testing
#filelist_damages=$filelist

echo
echo $filelist_damages
wc -l $filelist_damages
echo


output_dest=$exp_dir/matching_output/damagethresh${damage_thresh}
mkdir -p $output_dest

#interesting_filelist=$shakespeare_filelist
#candidates_filelist=$other_filelist

echo ""
#echo "**** NOTE: currently using combined filelist, instead of separate query/candidate lists"
echo ""

date

#exit 1

CUDA_VISIBLE_DEVICES=$d python3 ~/projects/git/anomaly-detection/matching/triplet_trainer_megabatch.py \
        --char $char \
        --model_type Attention \
        --batch_size $batch_size \
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
        --discover_matches "$filelist_queries" "$filelist_damages" \
        --discover_matches_split_amount $split_amount \
        --restrict_query_to $restrict_query_to \
        --query_shakespeare_part $shakespeare_part \
        --exclude_same_book_matches 
        #--exclude_book_name_from_candidates shakespeare4folio1685 \
        #--discover_matches "$filelist_damages" "$filelist_damages" \

        #--discover_matches "$interesting_filelist" "$candidates_filelist" \
date
