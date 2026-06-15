#!/bin/bash

# set -x

nr=5
ymd="2024-11-23"
# number of synthetic match pairs to create fake_match_eval_sets with
num_syn_match_pairs=20

# # hyperlink synthetic dataset to local directory
# local_link_dir=synthetic_data_linked
# mkdir -p $local_link_dir

# # first make the synthetic dataset fake_match_eval_sets
# cs="A B C D E F G H K L M N P Q R S T W"
# # c="B"
# for c in $cs; do
#     og_syn_char_dir="/trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-${c}-out/synthetic/${c}_twin_samebase_${nr}_${ymd}"
#     local_syn_char_dir=${local_link_dir}/${c}_twin_samebase_${nr}_${ymd}
#     python make_fake_match_eval_sets.py $og_syn_char_dir $og_syn_char_dir/fake_match_eval_sets --num_pairs $num_syn_match_pairs --splits train valid test
#     # link the synthetic dataset to a local directory for easier access/passing to matcher.py
#     # if [ -d $local_syn_char_dir ]; then
#     #     rm -rf $local_syn_char_dir
#     # fi
#     # ln -s $og_syn_char_dir $local_syn_char_dir
# done

# echo "Created fake_match_eval_sets for synthetic dataset"
# ls -l $local_link_dir
# echo

# echo "Creating leviathan preprocessed data"
# python preprocess_gt_test_set.py --test_set leviathan
python preprocess_gt_test_set.py --test_set lockespinoza