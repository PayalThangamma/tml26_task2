# Assignment 2: Stolen Model Detection

This repository implements a stolen-model detector for Assignment 2 using a combination of weight-based and behavior-based similarity signals.

The detector follows the tutorial intuition that different stealing methods leave different traces:

- Direct copies, checkpoint-derived models, and fine-tuned models leave traces in the weights.
- Knockoff or distilled models may have different weights but similar behavior on probe inputs.

Our method scores each of the 360 suspect models using:

- Global weight cosine similarity
- Global weight L2 similarity
- Logit cosine similarity on CIFAR-100 probe images
- Prediction agreement with the target model
- KL-divergence-based similarity
- Low-margin / hard-example logit similarity
- Low-margin prediction agreement

Best public leaderboard score reproduced:

```text
TPR@5%FPR = 0.462963
````

## Prerequisites

The following files and folders must be present in the repository root:

```text
target_model/
suspect_models/
task_template.py
submission.py
```

Expected model files:

```bash
find target_model -name "*.safetensors" | wc -l
find suspect_models -name "*.safetensors" | wc -l
du -sh .
```

Expected output:

```text
1
360
around 16G
```

The model files are not included in the CMS ZIP. They should be downloaded separately from the Hugging Face repository for Task 2.

A CUDA GPU is recommended. The provided Condor setup uses the following Docker image:

```text
pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

The CIFAR-100 dataset is downloaded automatically by `torchvision` into `./data` when `task_template.py` is run.

## Reproduce

### HTCondor

From the repository root:

```bash
cd tml26_task2
mkdir -p runlogs
```

### 1) Generate `submission.csv`

```bash
condor_submit task_template.sub
```

This runs `run_task_template.sh`, which executes `task_template.py`.

The script:

1. Loads the target CIFAR-style ResNet-18 model.
2. Loads the 360 suspect models.
3. Downloads/loads CIFAR-100 test images as probe data.
4. Computes weight-similarity and behavior-similarity features.
5. Writes:

   * `submission.csv`
   * `debug_scores.csv`

After the job finishes, check:

```bash
cat runlogs/task2.*.out
cat runlogs/task2.*.err
wc -l submission.csv
head submission.csv
```

Expected:

```text
361 submission.csv
```

The file should have the format:

```csv
id,score
0,0.2110027855153203
1,0.012813370473537604
...
```

### 2) Submit to the leaderboard

Before submitting, edit `submission.py` and set:

```python
API_KEY = "YOUR_API_KEY_HERE"
FILE_PATH = "submission.csv"
SUBMIT = True
```

Then submit using Condor:

```bash
condor_submit submission.sub
```

This runs `run_submit.sh`, which executes `submission.py` and uploads `submission.csv` to the leaderboard server.

Check the submission logs:

```bash
cat runlogs/submit.*.out
cat runlogs/submit.*.err
```

A successful upload prints:

```text
Successfully submitted.
```

## Quick sanity checks

Before running the scoring job, verify that the model files are real and not Git LFS pointer files:

```bash
du -sh .
find target_model -name "*.safetensors" | wc -l
find suspect_models -name "*.safetensors" | wc -l
```

The full folder should be around 16 GB. If it is only a few MB, the model files were not downloaded correctly.

After generating `submission.csv`, verify:

```bash
python - <<'PY'
import pandas as pd
import numpy as np

df = pd.read_csv("submission.csv")

print("shape:", df.shape)
print("columns:", list(df.columns))
print("id min/max:", df["id"].min(), df["id"].max())
print("unique ids:", df["id"].nunique())
print("missing ids:", sorted(set(range(360)) - set(df["id"])))
print("duplicate ids:", df["id"].duplicated().sum())
print("finite scores:", np.isfinite(df["score"]).all())
PY
```

Expected:

```text
shape: (360, 2)
columns: ['id', 'score']
id min/max: 0 359
unique ids: 360
missing ids: []
duplicate ids: 0
finite scores: True
```

## Files used to reproduce the result

| File                   | Role                                                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `task_template.py`     | Main scoring script. Loads the target and suspect models, computes similarity features, and writes `submission.csv` and `debug_scores.csv`. |
| `submission.py`        | Uploads `submission.csv` to the leaderboard server using the provided API key.                                                              |
| `run_task_template.sh` | Condor wrapper for running `task_template.py` inside the Docker container.                                                                  |
| `task_template.sub`    | Condor submit file for generating `submission.csv`.                                                                                         |
| `run_submit.sh`        | Condor wrapper for running `submission.py`.                                                                                                 |
| `submission.sub`       | Condor submit file for uploading `submission.csv`.                                                                                          |
| `README.md`            | Instructions to reproduce the result.                                                                                                       |
```

