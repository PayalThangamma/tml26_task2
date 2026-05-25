# Stolen-Model Detection — Assignment 2

Detects which of 360 suspect CIFAR-100 ResNet-18 checkpoints were derived from a known target model.

- **Metric:** TPR@5%FPR (higher is better)
- **Current leaderboard score:** `0.611111`
- **Detector entry point:** `task_template.py`
- **Leaderboard submitter:** `submission.py`

## Repository layout

```
tml26_task2/
├── task_template.py            # detector + scoring pipeline (produces submission.csv)
├── submission.py               # uploads submission.csv to the leaderboard
├── run_task_template.sh        # wrapper used by task_template.sub
├── run_submit.sh               # wrapper used by submission.sub
├── task_template.sub           # HTCondor submit file for scoring
├── submission.sub              # HTCondor submit file for leaderboard upload
├── target_model/
│   ├── weights.safetensors     # target ResNet-18 checkpoint
│   └── train_main_idx.json     # 40 000 indices of CIFAR-100 train samples used to train the target
├── suspect_models/             # 360 suspect checkpoints (.safetensors)
├── data/                       # CIFAR-100 (auto-downloaded by torchvision on first run)
├── submission.csv              # final scores (id,score) — produced by task_template.py
├── debug_scores.csv            # every per-suspect raw + ranked column (for inspection)
└── runlogs/                    # HTCondor output / error / log files
```

## How to reproduce the best result

### 1. Requirements

The scoring job runs in a `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` Docker container with one GPU. Extra Python packages needed:

```bash
pip install pandas safetensors requests
```

(All other dependencies — `torch`, `torchvision`, `numpy` — come from the base image.)

### 2. Expected inputs

Place the following before running:

- `target_model/weights.safetensors` — provided target checkpoint
- `target_model/train_main_idx.json` — provided target-training-index file
- `suspect_models/<i>/weights.safetensors` (or `suspect_models/<i>.safetensors`) for `i = 0..359`

CIFAR-100 is downloaded automatically into `./data/` on first run.

### 3. Run the scorer (HTCondor)

From the cluster head node:

```bash
condor_submit task_template.sub
```

Inspect progress (each suspect ≈ 4 s; full run ≈ 1.3 h on a P100):

```bash
condor_q
tail -f runlogs/task2.<cluster_id>.0.out
```

When the job finishes you will have a refreshed `submission.csv` and a `debug_scores.csv`.

### 3b. Run the scorer locally (optional)

```bash
python task_template.py
```

Same outputs.

### 4. Upload to the leaderboard

```bash
condor_submit submission.sub
```

(or `python submission.py` locally). The leaderboard enforces a ~6-minute cooldown between submissions.

## Configuration knobs

All knobs live at the top of `task_template.py`:

| Constant            | Default | Meaning                                                  |
|---------------------|---------|----------------------------------------------------------|
| `N_TEST_PROBE`      | 5000    | CIFAR-100 test images used per suspect                   |
| `N_TRAIN_PROBE`     | 5000    | target-train images (from `train_main_idx.json`)         |
| `N_NONTRAIN_PROBE`  | 5000    | non-target-train images (rest of CIFAR-100 train)        |
| `N_LOW_MARGIN`      | 300     | hard examples (lowest margin) used for focused similarity|
| `N_NOISE_PROBE`     | 512     | Gaussian-noise inputs for the OOD fingerprint            |
| `NOISE_SEED`        | 0       | RNG seed for the noise probe (fixed for reproducibility) |
| `TOPK_JACCARD`      | 5       | k for top-k label-set Jaccard agreement                  |
| `BATCH_SIZE`        | 128     | forward-pass batch size                                  |

Re-running with the defaults above reproduces the `submission.csv` that scored `0.611111`.

## Pipeline overview

`task_template.py` does the following:

1. Loads the target ResNet-18.
2. Builds three natural-image probe loaders (`test`, `target_train`, `non_target_train`) and a fixed Gaussian-noise probe tensor.
3. Computes target logits, softmax probs, avgpool features, predictions and labels on every probe.
4. For each of the 360 suspects:
   - Computes global and **layer-grouped** weight similarities (early / backbone / head / BN).
   - Runs the suspect on every probe and computes 9 behavioural similarities per split (logit cosine, KL, pred agreement, conf/loss correlation, avgpool feature cosine, hard-example similarity, top-5 Jaccard).
   - Runs the suspect on the Gaussian-noise probe (3 OOD signals).
5. Rank-normalises every raw signal across the 360 suspects.
6. Combines them into five composite detectors:
   - `weight_detector`        — pure parameter similarity
   - `behavior_detector`      — natural-image output similarity (test split)
   - `feature_detector`       — penultimate-layer features (test split)
   - `fingerprint_detector`   — target-training-subset signals + train/non-train gaps
   - `noise_detector`         — Gaussian-noise OOD response
7. Final score = `max` of `{weight, behavior+feature, behavior+fingerprint, noise, noise+behavior}` then rank-normalised and sharpened (`^1.15`).

The `max` is critical: TPR@5%FPR rewards the top 18 ranks, so a single strong signal should be enough to flag a stolen model.

## Files produced

- `submission.csv` — leaderboard format, two columns: `id`, `score`. 360 rows.
- `debug_scores.csv` — every raw column + every rank + every composite detector. Used to debug ranking changes.

## Results

The leaderboard metric is **TPR@5%FPR**, i.e. the fraction of stolen models that
end up in the top 5% of predicted scores (top 18 of 360). Three iterations of
the detector were submitted; the third is the current best.

| Iteration | Submission ID | Scoring job | Detector summary                                                                                          | TPR@5%FPR     |
|-----------|---------------|-------------|------------------------------------------------------------------------------------------------------------|---------------|
| Baseline  | 1711          | `47744`     | global weight cosine/L2 + test-split logit cosine, agreement, KL, hard-example similarity; `max` fusion    | low baseline  |
| It. 2     | 1778          | `47786`     | added grouped weights, features, train/non-train fingerprint, conf/loss correlation; trimmed-mean fusion   | regressed     |
| **Final** | **1793**      | **`47814`** | reverted to `max` fusion + added Gaussian-noise OOD probe and top-5 label Jaccard                          | **`0.611111`** |

Why the final iteration helps:

- **Gaussian-noise probe** adds a signal that is essentially uncorrelated with
  natural-image behaviour. Independent ResNet-18s have noise-logit cosines in
  `~0.1–0.3`; stolen / distilled copies stay close to `1.0`. This catches
  suspects that the weight, feature and behaviour detectors all rank in the
  middle of the pack.
- **`max` fusion** is the right operator for TPR@5%FPR: any single strong
  signal pushes a true stolen model into the top 18. The trimmed-mean variant
  tried in iteration 2 averaged correlated detectors and pulled distilled
  copies (low weight similarity, high behaviour) below the 5% FPR threshold.
- **Top-5 label-set Jaccard** adds rank information beyond `argmax` and is
  folded into both the behaviour and the training-fingerprint detectors.

Reproducing `0.611111` requires no changes — just run the steps in
*How to reproduce the best result* with the defaults in `task_template.py`.
