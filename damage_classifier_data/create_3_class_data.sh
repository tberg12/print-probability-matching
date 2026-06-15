echo "path,label" > 3_class_data.csv
# 0 - grab normals
find $PWD/filter_normals_redo_top_1pct_logprob_filter_class -name "*.tif" | sed -e 's/$/,0/g' >> 3_class_data.csv
# 1 - grab positive damages
while read -r line; do echo $PWD/dldt_ground_truth/imgs/$line,1; done < dldt_ground_truth/pos_filelist.txt >> 3_class_data.csv
# 2 - grab bad extractions
find $PWD/filter_bad_redo_lowest_1pct_logprob_filter_class -name "*.tif" | sed -e 's/$/,2/g' >> 3_class_data.csv

