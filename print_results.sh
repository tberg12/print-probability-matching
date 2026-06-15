echo Model: L2Baseline
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*L2Baseline*/log.log; do 
    echo $f;
    #grep 'Best' $f | tail -n 1
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
done

echo Model: RandomBaseline
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*randomBaseline*/log.log; do 
    echo $f;
    #grep 'Best' $f | tail -n 1
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
done

echo Model: Embed
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*__ModelL2Embedding*__Margin*Batchsz*/log.log; do 
    echo $f;
    #grep 'Best' $f | tail -n 1
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done

echo
echo Model: Embed-Jitter
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*__ModelL2Embedding*Jitter*Batchsz*/log.log; do 
    echo $f;
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done

echo
echo Model: Embed-Residual
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*TempResidual*ModelL2Embedding*__Margin*Batchsz*/log.log; do 
    echo $f;
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done

echo
echo Model: Embed-Residual-Jitter
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*TempResidual*ModelL2Embedding*Jitter*Batchsz*/log.log; do 
    echo $f;
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done

echo
echo Model: Attention-hw_sum
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*__ModelAttention*hw_sum*__Margin*/log.log; do 
    echo $f;
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done

echo
echo Model: Attention-hw_sum-Jitter
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*__ModelAttention*hw_sum*Jitter*/log.log; do 
    echo $f;
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done

echo
echo Model: Attention-hw_sum-Residual
#/trunk/nvog/matching_results_aaai2022/logs/*TempResidual*Attention*hw_sum*__Margin*Batchsz*/log.log
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*TempResidual*ModelAttention*hw_sum*__Margin*/log.log; do 
    echo $f;
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done

echo
echo Model: Attention-hw_sum-Residual-Jitter
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*TempResidual*ModelAttention*hw_sum*Jitter*/log.log; do 
    echo $f;
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done

echo
echo Model: Attention-filter_sum
for f in /trunk/nvog/matching_results_aaai2022/logs/Char*__ModelAttention*filter_sum*__Margin*/log.log; do 
    echo $f;
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done

echo Model: Attention-filter_sum-Residual
#CharT_TempResidual_ModelAttention_EmbSz128_hw_sum
for f in /trunk/nvog/matching_results_aaai2022_2/logs/Char*_TempResidual_ModelAttention*EmbSz128*hw_sum*__Margin*/log.log; do 
    echo $f;
    grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
    grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1
done
