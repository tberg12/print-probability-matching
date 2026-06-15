# Packaging plan — the original 444 GB `matching/` directory

This documents what was in Nikolai's working dir, how this clean repo was carved out of it,
and what to do with the ~370 GB left behind (keep / archive cold / drop). Generated from a
full inventory of `/graft2/code/nvog/git/matching` (444 GB, ~8M files).

## What this repo contains (Tier 1 — the clean hand-off, ~73 MB)

Assembled from the live filesystem with:
- all top-level `*.py` and `*.sh` (minus `secret.py`)
- `pyproject.toml`, `poetry.lock`
- `alignment_code/`, `Morpho-MNIST/`, `notebooks/` (caches/`.git` excluded)
- all `*.ipynb` **with outputs stripped** (181 MB notebook → 19 KB)
- `marimo/` scripts + small data tables (CSV/XLSX/JSON/HTML/`lib/`/`workbench_images/`);
  the ~45 MB of regenerable `*_latex_bigletter.png` figures and `.bkup` files were dropped
- `lists/` files ≤ 5 MB only (curated provenance; the ~4 GB of regenerable
  `faiss_image_paths_*` / `char_images3_paths.txt` manifests dropped)
- `damage_classifier_data/*.csv|*.sh|*.txt` (labeled training data; image dirs dropped)

Known gap: 5 `manual*` files in `lists/` were unreadable (perms) and must be backfilled —
see `HANDOFF.md` §4.4.

## Inventory & disposition of the full directory

| Cluster | Size | Disposition |
|---|---|---|
| Core Python (`matcher.py`, `models.py`, `datasets.py`, …) | 700 KB | **in this repo** |
| Shell scripts (34) | 268 KB | **in this repo** |
| `alignment_code/` + `Morpho-MNIST/` | 13 MB | **in this repo** |
| `marimo/` scripts + small tables | ~10 MB | **in this repo** |
| Notebooks (stripped) | <5 MB | **in this repo** |
| Curated `lists/` + damage label CSVs | ~36 MB | **in this repo** |
| GT test sets (leviathan, lockespinoza) | ~3.3 GB | **keep** — irreplaceable; too big for git |
| DLDT damage ground truth | 527 MB | **keep** — irreplaceable |
| Matcher `best.pt` checkpoints (35) | ~25 GB | **keep** — esp. the 2025-05-27 production ckpt |
| Damage classifier `resnet34 …3class/best.pt` | 256 MB | **keep** |
| `char_images3/` raw corpus | 81 GB | **archive cold** — re-fetchable from Bridges-2 |
| Matcher `epoch*.pt` snapshots | ~131 GB | **drop** — redundant given `best.pt` |
| FAISS indices (`.index`/`.npy`) | ~67 GB | **drop/regen** (keep 2–3 you'll query) |
| `data/synthetic_data/` generated training pairs | ~20 GB | **KEEP** (PI wants to reuse; do not regenerate) |
| `data/` other raw pools (ocean/, redo_*, tars) | ~50 GB | **archive/drop** — regenerable |
| `damage_classifier_output/` preds + abandoned ablations | ~29 GB | **drop** |
| `wandb/` | 5 GB | **compact** to per-run config/summary (~100 MB), then archive |
| `graph.log` (METAFONT error spew) + `graph.2602gf` | 540 MB | **drop** — accidental junk |
| `__MACOSX/`, `home/ppdevs/`, `*.tar.gz` dup archives, `*.pkl` graphs, `scratch/`, caches | ~6 GB | **drop** (verify `home/ppdevs/` page-scans have a canonical copy first) |

**Reclaimable now: ~300 GB** (epoch snapshots + FAISS + debug preds + dup archives + logs).
**New student's working set: ~30 GB** (code + GT data + best checkpoints) **plus 81 GB**
cold raw corpus.

## Suggested cleanup (NOT yet executed — review before running)

```bash
ROOT=/graft2/code/nvog/git/matching
# 1. Unambiguous junk (~1.6 GB):
rm -rf "$ROOT/__MACOSX" "$ROOT/__pycache__" "$ROOT/.ipynb_checkpoints" \
       "$ROOT/scratch" "$ROOT/graph.log" "$ROOT/graph.2602gf"
# 2. Redundant archives — ONLY after confirming the live dirs exist (~1.07 GB):
#    big_letters.tar.gz, damage_classifier_data.tar.gz, leviathan_*_preprocessed.tar.gz
# 3. Redundant epoch checkpoints (~131 GB) — keep each run's best.pt:
#    find "$ROOT/output" -name 'epoch*.pt' -delete   # review first!
# 4. Regenerable FAISS indices (~67 GB) — keep the few corpora you query.
# 5. Debug preds (~16 GB):  rm -rf "$ROOT/damage_classifier_output/evaluate/preds"
```
Verify each target before deleting; some "redundant" archives may be the only copy of a
dir that was later modified. `home/ppdevs/.../page_images` holds ~13k raw page scans — do
not delete until you confirm a canonical copy exists in the project's data store.
