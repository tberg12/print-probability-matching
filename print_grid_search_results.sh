

# TODO: check that the best results for these gt metrics are chosen based off the syn_valid stopping_metric!

#for f in /trunk/nvog/matching_results_aaai2022_2/logs/CharA_TempResidual_ModelAttention_*CNNBlocks_*ConvPerBlock*hw_sum*/log.log; do
#for f in /trunk/nvog/matching_results_aaai2022_2/logs/CharA_TempResidual_ModelAttention_*CNNBlocks_*ConvPerBlock*filter_sum*/log.log; do
#for f in /trunk/nvog/matching_results_aaai2022_2/logs/CharA_TempResidual_ModelAttention_2CNNBlocks_3ConvPerBlock*filter_sum*Margin{0.2,0.3,0.4,0.6,0.8}*/log.log; do
#for f in /trunk/nvog/matching_results_aaai2022_2/logs/CharA_TempResidual_ModelAttention_2CNNBlocks_3ConvPerBlock*filter_sum*Temp{0.2,0.4,0.6,0.8,1.0}*/log.log; do
model=Attention  #L2Embedding
residual=TempResidual
jitter="Temp1.0_Margin0.3"  #Jitter  #""
lds=5000
for split in 'syn_valid' 'pos' 'strong_neg' 'weak_neg' 'mix_neg'; do
    echo $split
    for f in /trunk/nvog/matching_results_aaai2022_WedAug10/logs/Char*_${residual}_Model${model}_${lds}TrainPairs_*CNNBlocks_*ConvPerBlock*filter_sum*${jitter}*/log.log; do
        echo $f
        grep '^\*\*\*\|>>>' $f | tail -n 5 | grep $split | grep -v 'Epoch' #| cut -d' ' -f4
        #grep -o "gt_test_${split}_recall_pct_[A-Z].5.*," $f | cut -d, -f1 #| cut -d' ' -f2
        #grep -o "best_syn_valid_recall_pct_[A-Z].5.*," $f | cut -d, -f1
        #grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
        #grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
        #grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    done
done
