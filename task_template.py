import os
import sys
import requests
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
import pandas as pd

# --------------------------------
# LOADING A MODEL (EXAMPLE: TARGET MODEL)
# --------------------------------

def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

checkpoint_path = "target_model/weights.safetensors"  # Replace with your model checkpoint path 
state_dict = load_file(checkpoint_path, device="cpu")

model = make_model() 
model.load_state_dict(state_dict, strict=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

data_root = "./data"  # Replace with your CIFAR-100 dataset path, or where it should be downloaded
dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
x, y = dataset[0]  # Example: get the first image and label

with torch.no_grad():
    logits = model(x.unsqueeze(0))

print("True label:", y)
print("Logits shape:", logits.shape)  # Should be [1, 100] for CIFAR-100
print("Logits:", logits)
   
# # --------------------------------
# # SUBMISSION FORMAT
# # --------------------------------

"""
The submission must be a .csv file with the following format:

-"id": ID of the subset (from 0 to 359)
-"score": Stealing confidence score for each model (float)
"""

# Example Submission:

subset_ids = list(range(360))  
confidence_scores = torch.rand(len(subset_ids)).tolist()
submission_df = pd.DataFrame({
    "id": subset_ids,
    "score": confidence_scores
})
submission_df.to_csv("submission.csv", index=None)
# --------------------------------
# REAL SCORING: overwrite random submission.csv
# --------------------------------

import glob
import re
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUSPECT_DIR = "suspect_models"
N_PROBE = 512
BATCH_SIZE = 128


def get_model_id(path):
    p = Path(path)

    # Try filename first: 0.safetensors, model_0.safetensors, suspect_123.safetensors
    m = re.search(r"(\d+)", p.name)
    if m:
        return int(m.group(1))

    # Try parent folder: suspect_models/123/weights.safetensors
    for part in reversed(p.parts):
        if part.isdigit():
            return int(part)

    raise ValueError(f"Could not extract model id from {path}")


def flatten_matching_params(sd1, sd2):
    xs = []
    ys = []

    for k in sorted(set(sd1.keys()) & set(sd2.keys())):
        a = sd1[k]
        b = sd2[k]

        if not torch.is_tensor(a) or not torch.is_tensor(b):
            continue

        if a.shape != b.shape:
            continue

        if a.numel() <= 1:
            continue

        xs.append(a.detach().float().cpu().reshape(-1))
        ys.append(b.detach().float().cpu().reshape(-1))

    if len(xs) == 0:
        return None, None

    return torch.cat(xs), torch.cat(ys)


def cosine_sim(a, b):
    return torch.dot(a, b).item() / ((a.norm().item() * b.norm().item()) + 1e-12)


def l2_sim(a, b):
    dist = torch.norm(a - b).item()
    denom = torch.norm(a).item() + torch.norm(b).item() + 1e-12
    return 1.0 - dist / denom


def rank_normalize(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 1:
        return np.ones_like(values)

    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    return ranks / (len(values) - 1)


def collect_logits(model, loader):
    all_logits = []
    all_preds = []

    model.eval()

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)

            all_logits.append(logits.detach().cpu())
            all_preds.append(logits.argmax(dim=1).detach().cpu())

    return torch.cat(all_logits), torch.cat(all_preds)


print("\nStarting real stolen-model scoring...")
print("Device:", DEVICE)

# Move target model to device
model = model.to(DEVICE)
model.eval()

# Probe subset from CIFAR-100
probe_indices = list(range(min(N_PROBE, len(dataset))))
probe_dataset = Subset(dataset, probe_indices)
probe_loader = DataLoader(
    probe_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Computing target logits...")
target_logits, target_preds = collect_logits(model, probe_loader)
target_probs = F.softmax(target_logits, dim=1)

target_sd = state_dict

suspect_files = sorted(
    glob.glob(os.path.join(SUSPECT_DIR, "**", "*.safetensors"), recursive=True)
)

print("Number of suspect models found:", len(suspect_files))

rows = []

for i, suspect_path in enumerate(suspect_files):
    model_id = get_model_id(suspect_path)
    print(f"[{i + 1}/{len(suspect_files)}] scoring suspect id {model_id}")

    suspect_sd = load_file(suspect_path, device="cpu")

    x_flat, y_flat = flatten_matching_params(target_sd, suspect_sd)

    if x_flat is None:
        weight_cos = -1.0
        weight_l2 = -1.0
    else:
        weight_cos = cosine_sim(x_flat, y_flat)
        weight_l2 = l2_sim(x_flat, y_flat)

    suspect_model = make_model()
    suspect_model.load_state_dict(suspect_sd, strict=True)
    suspect_model = suspect_model.to(DEVICE)
    suspect_model.eval()

    suspect_logits, suspect_preds = collect_logits(suspect_model, probe_loader)

    logit_cos = F.cosine_similarity(target_logits, suspect_logits, dim=1).mean().item()
    pred_agree = (target_preds == suspect_preds).float().mean().item()

    suspect_log_probs = F.log_softmax(suspect_logits, dim=1)
    kl = F.kl_div(suspect_log_probs, target_probs, reduction="batchmean").item()

    rows.append({
        "id": model_id,
        "weight_cos": weight_cos,
        "weight_l2": weight_l2,
        "logit_cos": logit_cos,
        "pred_agree": pred_agree,
        "neg_kl": -kl,
    })

    del suspect_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

if len(df) != 360:
    print("WARNING: expected 360 suspect models, found", len(df))

missing = sorted(set(range(360)) - set(df["id"].tolist()))
if missing:
    raise ValueError(f"Missing model ids: {missing[:20]}")

if df["id"].duplicated().any():
    raise ValueError("Duplicate model ids found")

for col in ["weight_cos", "weight_l2", "logit_cos", "pred_agree", "neg_kl"]:
    df[col + "_rank"] = rank_normalize(df[col].values)

df["score"] = (
    0.30 * df["weight_cos_rank"] +
    0.20 * df["weight_l2_rank"] +
    0.25 * df["logit_cos_rank"] +
    0.15 * df["pred_agree_rank"] +
    0.10 * df["neg_kl_rank"]
)

df["score"] = df["score"].clip(0.0, 1.0)

submission_df = df[["id", "score"]]
submission_df.to_csv("submission.csv", index=False)
df.to_csv("debug_scores.csv", index=False)

print("Saved real submission.csv")
print("Saved debug_scores.csv")
print(submission_df.head())
print(submission_df.tail())
