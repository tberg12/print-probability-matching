#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 6:00:00
#SBATCH --gpus=1


#echo commands to stdout
set -x
#
# Author: Kartikgo
# Date: Apr 16 2021
# Script Info:
# Runs the following steps for the line/character extraction workflow pipeline on a book:
#       * Stage 7: Untar and temp dir creation based on prepared csv for aligment
#	* Stage 8: Alignment of chars in the temp dir
# Run ./run_workflow2.sh for usage information.
#
# Prerequisites:
# run_workflow2.sh has successfully completed.
#Example: sbatch untar_align_slurm.sh /ocean/projects/hum160002p/shared/char_images3/  /ocean/projects/hum160002p/shared/char_images3_csvs/ L /ocean/projects/hum160002p/shared/char_images3_aligned/shakespeare/
# shakespeare current example:
# bash $KARTIKGO/untar_align_slurm.sh $SHARED/char_images3_alignment_input/shakespeare/tars/ $SHARED/char_images3_alignment_input/shakespeare/csvs/ M $SHARED/char_images3_aligned/shakespeare/shakespeare
#interesting: (8/19/2021)
# bash $OHOME/kartikgo/untar_align_slurm.sh $OHOME/kartikgo/pnp_alignment/alignment/exp/align_interesting/G/tars/ $OHOME/kartikgo/pnp_alignment/alignment/exp/align_interesting/G/csvs/ G $SHARED/char_images3_aligned/interesting/interesting

module load cuda/10.2.0
ANACONDA_ENVIRONMENT=/ocean/projects/hum160002p/kartikgo/anaconda3/bin
source $ANACONDA_ENVIRONMENT/activate
conda init
#conda init
#conda activate ~/anaconda3/envs/py3
TARFILES=${1%/}
CSVFILES="$2"
CHAR="$3"
PREFIX="$4"
LR="$5"
# move to working directory
cd /ocean/projects/hum160002p/nikolaiv/kartikgo/pnp_alignment/
##cd /trunk2/nvog/kartikgo/pnp_alignment/
#mkdir -p $4-$CHAR-$LR-out/char_"$CHAR"_uc
#python -u align_extract.py $TARFILES $CSVFILES $CHAR $4-$CHAR-$LR-out
#echo "Completed untarring and prepared for alignment"
which python
which python3
###for d in $(find /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-M-out/ocean -type d -name char_"$CHAR"_uc); do 
#for d in $(find $4-$CHAR-$LR-out -type d -name char_"$CHAR"_uc); do 
#    for f in $(find "$d" -name "*.tif"); do 
#        mv $f $4-$CHAR-$LR-out/char_"$CHAR"_uc/$(basename $f)
#    done
#done
#echo "CSV file count:"
#cut -d',' -f1 $4-$CHAR-$LR-out/"$CHAR"_align.csv | sort | uniq | wc -l
#echo "Actual file count:"
#find $4-$CHAR-$LR-out/char_"$CHAR"_uc -name "*.tif" > $4-$CHAR-$LR-out/char_"$CHAR"_uc_filelist.txt
#wc -l $4-$CHAR-$LR-out/char_"$CHAR"_uc_filelist.txt
bash aligner.sh $4-$CHAR-$LR-out $CHAR 0 $LR
echo "Completed alignment"
