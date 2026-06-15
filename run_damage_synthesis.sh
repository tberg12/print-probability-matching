# generate fake data for damage classifier training
python make_twin_dataset_from_splits.py \
    --input_dir /graft2/code/nvog/git/matching/data/redo_top_5pct_logprob \
    --output_dir /graft2/code/nvog/git/matching/data/synthetic_data/redo_top_5pct_logprob_singleton_5_2024-11-04 \
    --seed 42 \
    --binarize_image \
    --num_jobs 4 \
    --train_size 100000 \
    --valid_size 10000 \
    --test_size 10000