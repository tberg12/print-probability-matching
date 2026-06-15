# Print & Probability — Character/Type-Imprint Matching

Code for the **matching** stage of the [Print & Probability](http://printprobability.org)
project: attributing anonymously-printed early-modern English books (c. 1500–1800) by
matching the imprints of **uniquely damaged pieces of metal type** across books.

This repository was assembled from the working directory of **Nikolai Vogler** (PhD, UCSD),
who built and ran this pipeline, as a clean hand-off for a new student. The code is the
intellectual core; the large data, image corpora, and trained model checkpoints are **not**
in git (they are far too large) — see [`HANDOFF.md`](HANDOFF.md) for exactly where each
lives and how to fetch it.

> **Original author:** Nikolai Vogler (`nvog`). Please preserve attribution.
> **Project PIs:** Taylor Berg-Kirkpatrick (UCSD CSE) & Christopher Warren (CMU).

---

## The idea in one paragraph

Each piece of movable metal type acquires unique physical damage (bends, fractures,
wear). When inked and pressed it leaves a distinctive **type-imprint** on the page. If the
same damaged letter appears in two books, those books were (probably) printed by the same
shop with the same type — which lets us attribute books that were published anonymously to
evade censorship. The pipeline learns an embedding in which two imprints of the *same
damaged type* are close and everything else is far, then does large-scale nearest-neighbor
retrieval to surface candidate matches for a human bibliographer to confirm.

## Pipeline (scan → attribution)

```
scanned page
  → Ocular OCR segments & classifies characters            (upstream, not in this repo)
  → ALIGN each glyph to a canonical template                alignment_code/
  → DAMAGE CLASSIFIER keeps only damaged candidates         damage_classifier.py
  → MATCHER embeds each imprint (metric learning)           matcher.py + models.py
  → FAISS nearest-neighbor retrieval over embeddings        matcher.py --faiss_*
  → ranked matches → graph / voting analysis                marimo/
  → printer attribution + paper figures
```

Training labels are scarce, so positives are **synthesized**: take two undamaged imprints
and inject realistic bends / fractures / over- & under-inking (vendored, extended
`Morpho-MNIST/`).

## Two model generations (important)

- **AAAI-2023 CAML** — a shallow CNN that subtracts a per-letter average *template* to
  isolate damage, then an attention map over the residual produces the embedding
  (`models.py:AttentionNet`). This is the *published* model.
- **2024–2025 ViT dual-encoder + FAISS** — the newer, **uncommitted** direction Nikolai
  actually ended on: a `timm` ViT-B/16 dual-encoder trained with a CLIP-style contrastive
  loss, retrieved at scale with FAISS. **This is the current production model.**

**Production recipe** (best run, `2025-05-27`): `DualEncoder`, encoder
`hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`, 16 capital letters
(A B C D E F G H L M N P R S T W), `clip_extra` loss, margin 0.3, `add_global_inking`,
Adam lr 1e-4, linear-warmup+cosine, batch 224, 30k synthetic train pairs.
Best metric: **`gt_test_mix_neg` recall@5 = 83.9%** at epoch 26.
The checkpoint and `build_faiss_index.sh`/`query_faiss_index.sh` both point at
`output/matching_results_may25/CharABCDEFGHLMNPRSTW_ModelDualEncoder_..._2025-05-27_00:36:08/best.pt`.

## Repository layout

| Path | What |
|---|---|
| `matcher.py` | **Main entrypoint** — train / evaluate-ckpt / discover-matches / FAISS build+query |
| `models.py` | Model zoo: DualEncoder, CrossEncoder, AttentionNet (CAML), DamageCNN, ViT, … |
| `datasets.py` | Twin/synthetic + ground-truth datasets; on-the-fly damage augmentation |
| `damage_classifier.py` | Train/eval/predict the damaged-vs-normal-vs-bad-extraction classifier |
| `losses.py`, `matching_losses.py`, `supconloss.py`, `attention.py`, `model_args.py` | Model internals & data-derived constants |
| `*.sh` | The runbook — every training/eval/discover/FAISS command, with exact hyperparameters |
| `alignment_code/` | Glyph→template registration (PyTorch generative aligner) |
| `Morpho-MNIST/` | Vendored + extended synthetic-damage generator (`character_bender.py`, `perturb.py`) |
| `marimo/` | Final attribution analysis: book-match network + LaTeX figures (Nikolai's most recent work) |
| `notebooks/`, `*.ipynb` | Results viewers (output-stripped) — start at `discover_show_results_faiss.ipynb` |
| `lists/` | Curated file-path manifests / provenance (large regenerable ones omitted) |
| `damage_classifier_data/*.csv` | Labeled damage-classifier training data |

## Quickstart

```bash
poetry install            # env from pyproject.toml / poetry.lock (Python 3.9)
# point the scripts at your data/checkpoints, then e.g.:
bash train_matcher_unaligned.sh        # train the matcher
bash build_faiss_index.sh              # embed a corpus into a FAISS index
bash query_faiss_index.sh              # retrieve nearest type-imprints
```

## ⚠️ Before anything runs

Scripts, test-set CSVs, and `lists/` manifests contain **hard-coded absolute paths**
(`/graft2/...`, `/trunk2/nvog/shared/...`, `/home/nvog/...`, and PSC Bridges-2
`/ocean/projects/hum160002p/...`). These must be rewired to your environment. The big data
and model checkpoints are **not in this repo** — see [`HANDOFF.md`](HANDOFF.md) for their
locations and the full known-issues list.
