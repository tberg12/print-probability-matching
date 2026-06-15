bl=random # L2
for split in 'syn_valid' 'pos' 'strong_neg' 'weak_neg' 'mix_neg' 'areo'; do
    echo $split
    for f in /trunk/nvog/matching_results_aaai2022_ThursAug11/logs/Char*${bl}Baseline*/log.log; do
        echo $f;
        #grep 'Best' $f | tail -n 1
        grep -o "best_gt_test_pos_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
        grep -o "best_gt_test_strong_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
        grep -o "best_gt_test_weak_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
        grep -o "best_gt_test_mix_neg_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
        grep -o "best_gt_test_areo_recall_pct_[A-Z].5.*," $f | cut -d, -f1 | tail -n 1
    done | grep $split | cut -d' ' -f2
done
