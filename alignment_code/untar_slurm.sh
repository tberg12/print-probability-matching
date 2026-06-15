#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 1:00:00

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
module load cuda/10.2.0
ANACONDA_ENVIRONMENT=/ocean/projects/hum160002p/kartikgo/anaconda3/bin
source $ANACONDA_ENVIRONMENT/activate
conda init
TARFILES=${1%/}
CSVFILES="$2"
CHAR="$3"
PREFIX="$4"
# move to working directory
cd /ocean/projects/hum160002p/kartikgo/pnp_alignment/
python align_extract.py $TARFILES $CSVFILES $CHAR $4-$CHAR-out
echo "Completed untarring and prepared for alignment"
#bash aligner.sh $4-$CHAR-out $CHAR 0
#echo "Completed alignment"
