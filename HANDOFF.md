# Hand-off notes — reviving Nikolai's matching pipeline

Audience: the new student taking over the **matching** stage of Print & Probability.
Read [`README.md`](README.md) first for the high-level picture; this doc covers the
operational detail: how to run it, where the (gitignored) data and models live, and the
landmines.

---

## 0. Provenance / the most important thing to know

The original working directory was **`/graft2/code/nvog/git/matching`** (444 GB,
~8 million files) on Nikolai's account (`nvog`). Its **git history is stale** — only 7
commits, last one April 2025 — and most of the live code (incl. all of `marimo/`, dated
Dec 2025–Jan 2026) was **never committed**. This repository was assembled from the live
filesystem, not from GitHub. The 444 GB original still exists and is the source of truth
for the data/checkpoints listed below.

## 1. Environment

```bash
poetry install          # pyproject.toml pins Python ^3.9; faiss-cpu, torch, torchvision, timm, …
```
Wandb is used for logging (`wandb.init(project="matching2025", entity="nvog")`); set your
own entity or disable with the `--wandb` flag off. `Morpho-MNIST/morphomnist` must be on
`PYTHONPATH` for synthetic-data generation (the scripts `sys.path.append` it).

## 2. End-to-end run order

| Stage | Script → entrypoint | Notes |
|---|---|---|
| Synthesize training pairs | `run_same_base_damage_synthesis.sh` → `make_twin_dataset_from_splits.py` | uses `Morpho-MNIST` damage injection |
| Train damage classifier | `train_damage_classifier.sh` → `damage_classifier.py` | resnet34, 3-class |
| Predict damage (filter) | `run_damage_detection_prediction.sh` | SLURM job; gates which chars feed the matcher |
| Build GT eval sets | `preprocess_match_test_sets.sh` → `preprocess_gt_test_set.py` | leviathan / lockespinoza |
| **Train matcher** | `train_matcher_unaligned.sh` → `matcher.py` | the current ViT dual-encoder |
| **Build FAISS index** | `build_faiss_index.sh` → `matcher.py --faiss_build_index` | one `setting` per corpus |
| **Retrieve / attribute** | `query_faiss_index.sh` or `discover_matcher_unaligned_dualenc_may25.sh` | top-k nearest imprints |
| Evaluate | `eval_matcher_unaligned.sh` | recall@k on `gt_test_mix_neg` |
| Analyze / figures | `marimo/viz_votes.py`, `marimo/graph.py`, `discover_show_results_faiss*.ipynb` | |

`matcher.py` is a single argparse program; the four modes are selected by
`--evaluate_ckpt` / `--discover_matches` / `--faiss_build_index` / `--faiss_query_set_file`
(see `main()` and the `__main__` block). In `build_faiss_index.sh` / `query_faiss_index.sh`
the active corpus is whichever `setting=` line is left **uncommented**.

## 3. Where the data & models live (NOT in this repo)

All paths below are under the original `/graft2/code/nvog/git/matching/` unless noted.

### Critical — irreplaceable, cannot be regenerated
- **Ground-truth matching test sets** — `data/leviathan_matching_test_set_preprocessed/`
  (2.6 GB) and `data/lockespinoza_matching_test_set(_preprocessed)/` (~0.7 GB). Expert
  bibliographer labels; these are `matcher.py`'s default `--gt_data_dir` /
  `--lockespinoza_data_dir`. **The image paths inside their `matches.csv` /
  `*_negative_background_set.txt` are absolute and stale** (`/home/nvog/projects/git/...`)
  and must be rewritten.
- **Damage-classifier ground truth** — `damage_classifier_data/dldt_ground_truth/` (527 MB)
  + the label CSVs (those CSVs *are* included in this repo under `damage_classifier_data/`).
- **`marimo/` hand-curated tables** — `LockeSpinozaMatches_processed.csv`,
  `votes_map_nikolai.csv`, `votes_issue_clusters_chris.csv`, the `sample_review*.xlsx`
  and per-book `*map*.xlsx`. Human match judgments / expert annotations; **included in
  this repo** (small) — no code regenerates them.

### Critical — expensive to reproduce (keep, don't bundle in git)
- **Production matcher checkpoint** — `output/matching_results_may25/CharABCDEFGHLMNPRSTW_ModelDualEncoder_..._2025-05-27_00:36:08/best.pt` (~1 GB). The single most important artifact.
- **All `best.pt`** across runs (~25 GB total; 35 files). The `epoch*.pt` snapshots
  (~131 GB) are redundant once you have each run's `best.pt`.
- **Damage classifier** — `damage_classifier_output/resnet34_frompretrained_lr0.0001_real_3class_outputs/best.pt` (256 MB), the deployed model.

### Critical — raw corpus (archive cold; re-fetchable)
- **`char_images3/`** (81 GB, 1,300 per-book `.tar` of cropped per-letter imprints). The
  substrate everything matches over. Mirrors PSC Bridges-2
  `/ocean/projects/hum160002p/shared/char_images3/`; re-fetchable via the sftp workflow
  documented in `char_images3/how_to_sftp.txt` + the `*_bridges.txt` pull-lists.

### Regenerable (safe to drop / rebuild on demand)
- FAISS indices `output/faiss_index_*.index|.npy` (~67 GB) — rebuild from prod ckpt + char images.
- `data/synthetic_data/` image dirs (~20 GB), `data/ocean/`, `data/redo_*` tars (~70 GB).
- `lists/faiss_image_paths_*.txt`, `char_images3_paths.txt` (the multi-hundred-MB ones) —
  regenerate with the `find $PWD/<dir> -name '*.tif' > lists/<name>.txt` one-liners in the
  build scripts. (The small curated `lists/` files are included here.)
- `damage_classifier_output/evaluate/preds/` (16 GB debug image copies).

## 4. Known issues / landmines

1. **Hard-coded absolute paths** everywhere — `/graft2/...`, `/trunk2/nvog/shared/...`,
   `/home/nvog/...`, PSC Bridges `/ocean/projects/hum160002p/...`. Grep and rewrite before
   running. The GT test-set CSVs and `lists/` manifests are the worst offenders.
2. **`losses.py` defines `TripletDualEncoderLoss` twice** — the second definition shadows
   the first. Confirm you're using the intended one.
3. **`areopagitica_matching_test_set`** is referenced as a default in `matcher.py` but is
   **absent** from `data/` — locate or regenerate it, or override the flag.
4. **5 `manual*` provenance files in `lists/`** (`manuallocke_fnames.csv`,
   `manualspinoza_fnames.csv`, `manualfortysermons.txt`, `manualfortysermons_fnames.csv`,
   `manualcriticalenquiries_fnames.csv`) were mode `-rw-r-----` owned by `nvog` and could
   not be read when this repo was assembled — **they are missing here** and need to be
   backfilled (chmod/sudo from the original) to fully reproduce the Locke/Spinoza/Forty
   Sermons case studies.
5. **SLURM assumptions** — several scripts (`run_damage_detection_prediction.sh`, the
   `alignment_code/*slurm*.sh`) carry PSC Bridges-2 `#SBATCH` headers and cluster module
   loads; rewrite for your scheduler.
6. **`marimo/` is misnamed** — these are plain `python viz_*.py` scripts (polars/altair/
   networkx/pyvis), **not** marimo notebooks. Run them directly.
7. **`--evaluate_ckpt` throws a benign `UnboundLocalError` at the very end** — after all
   metrics are computed and printed, `train_model()` runs
   `print(f'Finished training ... {best_metric_name} ...')`, but `best_metric_name` is only
   assigned when a new best checkpoint is saved during training, which never happens in
   eval-only mode. The recall numbers are fully valid; the traceback is cosmetic. Fix:
   guard that print on `if not args.evaluate_ckpt:` (or initialize `best_metric_name`).
8. **Several scripts write to the *current directory*** (e.g. `datasets.py` saves
   `negative-hist-{split}.png` via `savefig` with a relative path). If you run from a
   directory you can't write to (like the original `nvog`-owned tree), the run dies with a
   `PermissionError`. **Run from a writable working dir** (e.g. a clone of this repo with the
   data dirs symlinked in), and point `--output_dir` / `--log_dir` / `--evaluate_ckpt_dest`
   at paths you own. See `RESULTS.md` for a working eval invocation that does exactly this.

## 5. Best places to start reading

- `matcher.py` `main()` + `__main__` — the four run modes.
- `train_matcher_unaligned.sh`, `build_faiss_index.sh`, `query_faiss_index.sh` — the
  current production invocation and the exact checkpoint path.
- `discover_show_results_faiss.ipynb` — cleanest results viewer.
- `marimo/viz_votes.py` — the most recent analysis (the anonymous "Votes of the House of
  Commons 1690" attribution, overlapping damaged-type matches with printers
  everingham / tbraddyll).
- The W&B run `wandb/latest-run` (in the original dir) documents the winning recipe.
