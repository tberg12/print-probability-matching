# Eval results — matcher checkpoints (reproduced 2026-06-15)

Reproduction of the matcher's retrieval recall on the **Leviathan ground-truth** matching
set (`gt_test_mix_neg`, 15 capital letters with GT data), run from this repo against the
checkpoints in the original working dir, on one RTX A6000. Metric = % of query type-imprints
whose true match is retrieved within top-k, averaged over letters ("allchars").

## Headline — production checkpoint reproduces exactly

The deployed model (`…2025-05-27…/best.pt`: DualEncoder, ViT-B/16 augreg2, `clip_extra`,
30k synthetic train pairs) at the paper's eval setting (background = 50 distractors/letter):

| | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| **gt_test_mix_neg** | 83.3 | 82.3 | **83.9** | 85.9 |

Matches the reported **recall@5 = 83.9%** to the decimal. Per-letter R@5: N 100 · D 93.3 ·
F 92.9 · P 92.5 · E 92.3 · R 87.8 · C 87.2 · W 85.7 · H 81.2 · G 79.7 · M 79.3 · B 75.9 ·
T 75.4 · L 68.8 · A 66.7. (Leviathan has no **S** GT data.)

## Checkpoint sweep (`may25`, background = 50)

| Model | Encoder | Loss | Train pairs | R@1 | R@5 | R@10 |
|---|---|---|---|---:|---:|---:|
| **DualEncoder** ⭐ prod | ViT-B/16 augreg2 | clip_extra | 30k | 83.3 | **83.9** | 85.9 |
| DualEncoder | ViT-B/16 augreg2 | clip_extra | 15k | 84.2 | **83.9** | 86.7 |
| DualEncoder | ViT-B/8 dino | clip_extra | 15k | 81.5 | **83.9** | 84.7 |
| DualEncoder | ViT-B/8 dino | clip_extra | 15k | 81.9 | 83.7 | 86.7 |
| DualEncoder | ViT-B/16 augreg2 | clip_extra | 15k | 82.2 | 82.6 | 85.1 |
| DualEncoder | ConvNeXt-B | clip_extra | 15k | 80.6 | 82.3 | 85.7 |
| DualEncoder | ViT-B/8 dino | npairs | 15k | 79.8 | 81.9 | 84.9 |
| DualEncoder | ConvNeXt-B | npairs | 15k | 67.6 | 67.5 | 71.7 |
| CrossEncoder | ViT-B/16 augreg2 | triplet_bce | 15k | 47.7 | 53.6 | 61.8 |
| DualEncoder | ViT-B/8 dino | npairs | 15k | 53.8 | 52.4 | 57.1 |
| DualEncoder | ViT-B/8 dino | npairs | 15k | 55.1 | 52.3 | 55.1 |
| DualEncoder | ViT-B/16 augreg2 | npairs | 15k | 53.4 | 51.8 | 56.1 |
| DualEncoder | ViT-B/8 dino | npairs | 15k | 12.2 | 18.3 | 25.1 |
| CrossEncoder | ViT-B/16 augreg2 | triplet_bce | 15k | 16.7 | 18.5 | 24.8 |

### Harder setting — production checkpoint, background = 1000

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| **gt_test_mix_neg (bg=1000)** | 75.3 | **72.4** | 74.0 |

20× more distractors per letter ⇒ ~11.5-point R@5 drop (83.9 → 72.4). This is the more
realistic large-corpus retrieval difficulty.

## Takeaways

- **`clip_extra` loss is the key win.** Every `clip_extra` DualEncoder scores 82–84% R@5;
  switching to `npairs` collapses most runs to ~52% (and `npairs` training is unstable —
  the ViT-B/8 dino npairs runs range 18 → 82). This is the single biggest factor.
- **Encoder choice barely matters once loss is fixed.** ViT-B/16, ViT-B/8 dino, and
  ConvNeXt-B with `clip_extra` all land 82–84% R@5. The chosen ViT-B/16 / 30k-pairs model is
  a sensible production pick but is statistically tied with several 15k-pair variants at
  bg=50; it likely earns its place on robustness at harder settings.
- **CrossEncoders are not worth it here** — far lower recall (18–54%) *and* dramatically
  slower at eval (a ViT forward per query×candidate pair). DualEncoder + FAISS is correct.

## How to reproduce

Run from a **writable** working dir (this repo, with the read-only data symlinked in) so
stray `savefig`/log writes succeed; redirect all outputs to a dir you own; reach the
checkpoint through a flat one-component symlink so `--load_model`'s parent name doesn't
contain a `/` (see `HANDOFF.md` §4.7–4.8 for the two bugs this avoids).

```bash
# from ~/print-probability-matching, with data/ char_images3/ damage_classifier_output/ symlinked in
RUN=~/pp_eval_runs; mkdir -p "$RUN/prod_ckpt"
ln -sfn <ORIG>/output/matching_results_may25/<prod-run>/best.pt "$RUN/prod_ckpt/best.pt"
CUDA_VISIBLE_DEVICES=7 <ORIG>/.venv/bin/python matcher.py \
  --char A B C D E F G H L M N P R S T W --margin 0.3 \
  --synthetic_data_dir data/synthetic_data/normal_clf_data \
  --model_type DualEncoder \
  --encoder_type hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --train_batch_size 224 --optimizer Adam --amp --use_pretrained \
  --loss_type clip_extra --negative_mining random_half_normal \
  --output_dir "$RUN" --log_dir "$RUN/logs" \
  --limit_dataset_size 1500 --limit_eval_dataset_size 50 \
  --limit_synthetic_eval_set_size 50 --limit_gt_eval_background_set_size 50 \
  --load_model prod_ckpt/best.pt \
  --evaluate_ckpt_dest "$RUN/evaluate_ckpt" --evaluate_ckpt \
  --k 1000 --limit_eval_keys gt_test_mix_neg
```

Background-50 reproduces the 83.9% headline; set the three `--limit_*` to 1000 for the
harder bg=1000 number. The full sweep harness is `~/pp_sweep.sh`.
