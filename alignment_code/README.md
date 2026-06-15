
# How to run alignment

## aligning with run_workflow2_eynollah.sh

### prep

this dir has example char_images3_alignment_input structure for prep script: 
ALIGNIN=/trunk2/nvog/shared/char_images3_alignment_input/shakespeare
ALIGNOUT=/trunk2/nvog/shared/char_images3_aligned/shakespeare_2024-10-26

```bash
CHARS=(A B C D E F G H K L M N P Q R S T W)
SHARED_DIR=/ocean/projects/hum160002p/shared
ALIGNMENT_DIR=/ocean/projects/hum160002p/nikolaiv/kartikgo
ALIGNIN=$SHARED_DIR/char_images3_alignment_input/$book_name
ALIGNOUT=$SHARED_DIR/char_images3_aligned/$book_name
save_prefix=$ALIGNOUT/char
lr=0.1

for c in ${CHARS[@]}; do
    # alignment ckpt
    ckpt=$SHARED_DIR/char_images3_aligned/shakespeare_2021-09-28/shakespeare-${c}-out/${c}_model/model_$(find $SHARED_DIR/char_images3_aligned/shakespeare_2021-09-28/shakespeare-${c}-out/${c}_model -name "*.ckpt" | xargs -L1 -I{} basename "{}" .ckpt | cut -d'_' -f2 | sort -nr | head -1).ckpt
    echo "Status: Prepping alignment for ${c}"
    echo sbatch $ALIGNMENT_DIR/prep_untar_alignstretch_slurm_train_from_ja.sh $ALIGNIN/tars/ $ALIGNIN/csvs/ $c $save_prefix $ckpt $lr
    sbatch -W -t 1:00:00 $ALIGNMENT_DIR/prep_untar_alignstretch_slurm_train_from_ja.sh $ALIGNIN/tars/ $ALIGNIN/csvs/ $c $save_prefix $ckpt $lr &
done
```

### align
```bash
for c in ${CHARS[@]}; do
    # alignment ckpt
    ckpt=$SHARED_DIR/char_images3_aligned/shakespeare_2021-09-28/shakespeare-${c}-out/${c}_model/model_$(find $SHARED_DIR/char_images3_aligned/shakespeare_2021-09-28/shakespeare-${c}-out/${c}_model -name "*.ckpt" | xargs -L1 -I{} basename "{}" .ckpt | cut -d'_' -f2 | sort -nr | head -1).ckpt
    echo "Status: Aligning ${c} using checkpoint at $ckpt"
    echo sbatch $ALIGNMENT_DIR/untar_alignstretch_slurm_train_from.sh $ALIGNIN/tars/ $ALIGNIN/csvs/ $c $save_prefix $ckpt $lr
    #sbatch -W -t 2:00:00 $ALIGNMENT_DIR/untar_align_slurm_train_from.sh $ALIGNIN/tars/ $ALIGNIN/csvs/ $c $save_prefix $ckpt $lr &
    sbatch -W -t 1:00:00 $ALIGNMENT_DIR/untar_alignstretch_slurm_train_from.sh $ALIGNIN/tars/ $ALIGNIN/csvs/ $c $save_prefix $ckpt $lr &
done
```


```bash
CHARS=(A B C D E F G H K L M N P Q R S T W)
SHARED_DIR=/trunk2/nvog/shared
ALIGNMENT_DIR=/home/nvog/projects/git/damaged-type/detect_and_match/alignment_code
ALIGNIN=/trunk2/nvog/shared/char_images3_alignment_input/shakespeare
ALIGNOUT=/trunk2/nvog/shared/char_images3_aligned/shakespeare_2024-10-26
save_prefix=$ALIGNOUT/char
lr=0.1

for c in ${CHARS[@]}; do
    # alignment ckpt
    ckpt="placeholder"
    echo "Status: Prepping alignment for ${c}"
    bash $ALIGNMENT_DIR/prep_untar_alignstretch_slurm_train_from_ja.sh $ALIGNIN/tars/ $ALIGNIN/csvs/ $c $save_prefix $ckpt $lr
done
```

```bash
c=S
lr=0.1
epochs=3000
lbd=0.05  # l2 regularization on templates
ink_var=0.2
y_pix=64
batch_size=32
data="${ALIGNOUT}/char-${c}-${lr}-out"

CUDA_VISIBLE_DEVICES=0 python pnp_alignment/alignment/aligner_stretch.py \
    -ink_var $ink_var \
    -data "$data" \
    -aligncsv "${c}_align.csv" \
    -init_avg \
    -nc 1 \
    -y_pix "$y_pix" \
    -gpu 0 \
    -epochs "$epochs" \
    -optim adam \
    -adj_learning_rate "$lr" \
    -visualize \
    -batch_size "$batch_size" \
    -output "${c}_template" \
    -rec "${c}_aligned" \
    -save_dir "${c}_model" \
    -lbd "$lbd" \
    -no_shear
```
