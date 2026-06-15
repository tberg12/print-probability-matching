#!/bin/bash
KARTIKGO=/ocean/projects/hum160002p/nikolaiv/kartikgo
SHARED=/ocean/projects/hum160002p/shared

set -x
# test with A first
for c in A; do
#for c in B C D E F G H K L M N P Q R S T W; do
    sbatch $KARTIKGO/untar_align_slurm.sh $SHARED/char_images3_alignment_input/shakespeare/tars/ $SHARED/char_images3_alignment_input/shakespeare/csvs/ $c $SHARED/char_images3_aligned/shakespeare/shakespeare
done
ligned/shakespeare/shakespeare
done
