#!/bin/bash
#SBATCH -c 7
#SBATCH --mem-per-cpu 1999MB
#SBATCH -p RM-shared
#SBATCH -t 2:00:00

source ~/.bashrc
module load anaconda3
ANACONDA_ENVIRONMENT=/ocean/projects/hum160002p/kartikgo/anaconda3/bin
conda activate "$ANACONDA_ENVIRONMENT"
#conda init
#conda activate ~/anaconda3/envs/py3
TARFILES=${1%/}
CSVFILES="$2"
CHAR="$3"
PREFIX="$4"
TRAINFROM="$5"
LR="$6"
# move to working directory
cd /ocean/projects/hum160002p/nikolaiv/kartikgo/pnp_alignment/
#cd /trunk2/nvog/kartikgo/pnp_alignment/
mkdir -p $4-$CHAR-$LR-out
python align_extract.py $TARFILES $CSVFILES $CHAR $4-$CHAR-$LR-out
echo "Completed untarring and prepared for alignment"
which python
which python3
#for d in $(find /trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-M-out/ocean -type d -name char_"$CHAR"_uc); do 
mkdir -p $4-$CHAR-$LR-out/char_"$CHAR"_uc
for d in $(find $4-$CHAR-$LR-out/ocean -type d -name char_"$CHAR"_uc); do 
    for f in $(find "$d" -name "*.tif"); do 
        mv $f $4-$CHAR-$LR-out/char_"$CHAR"_uc/$(basename $f)
    done
done
echo "CSV file count:"
cut -d',' -f1 $4-$CHAR-$LR-out/"$CHAR"_align.csv | sort | uniq | wc -l
echo "Actual file count:"
find $4-$CHAR-$LR-out/char_"$CHAR"_uc -name "*.tif" > $4-$CHAR-$LR-out/char_"$CHAR"_uc_filelist.txt
wc -l $4-$CHAR-$LR-out/char_"$CHAR"_uc_filelist.txt


