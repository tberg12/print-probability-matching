#!/bin/bash
set -x

export OHOME=/ocean/projects/hum160002p/nikolaiv
export SHARED=/ocean/projects/hum160002p/shared/
for lr in 0.1; do for c in G; do sbatch $OHOME/kartikgo/untar_alignstretch_slurm_train_from.sh $SHARED/char_images3_alignment_input/shakespeare_2021-09-28/align_kingsloo_2022-01-11/${c}/tars/ $SHARED/char_images3_alignment_input/shakespeare_2021-09-28/align_kingsloo_2022-01-11/${c}/csvs/ ${c} $SHARED/char_images3_aligned/shakespeare_2021-09-28/align_kingsloo_2022-01-11/shakespeare-testoriggutter $SHARED/char_images3_aligned/shakespeare_2021-09-28/shakespeare-${c}-out/${c}_model/model_$(find $SHARED/char_images3_aligned/shakespeare_2021-09-28/shakespeare-${c}-out/${c}_model -name "*.ckpt" | xargs -L1 -I{} basename "{}" .ckpt | cut -d'_' -f2 | sort -nr | head -1).ckpt $lr; done; done
