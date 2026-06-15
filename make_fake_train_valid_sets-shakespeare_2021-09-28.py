#c=Q
#d=0
#epochs=40
#jitter="--jitter"
#loss=object_ce
#
#exp=shakespeare4foliopart1
#data_dir=experiments/$exp/$c/detector_output
#mkdir -p $data_dir
#train=$data_dir/train_data.csv
#valid=$data_dir/valid_data.csv
#test=$data_dir/test_data.csv
#output=$data_dir
#CUDA_VISIBLE_DEVICES=$d python3 damage_classifier.py $train $valid $test $output --num_epochs $epochs --best_metric p@r=0.75 $jitter --loss_type object_ce --output_pooling max_pool_f # > damage_classifier_output/shakespeare_test_on_interesting/${c}/jitter.log



