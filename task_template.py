"""Stolen-model detector for Assignment 2 (CIFAR-100 ResNet-18 stealing).

Pipeline overview
-----------------
For each of the 360 suspect checkpoints we compute a *stealing confidence* score
that maximises ``TPR@5%FPR``. Only the relative ranking of the highest-scored
suspects matters for that metric, so the detector is designed around the
principle:

    *any single strong similarity signal should be enough to push a true*
    *stolen model into the top 5% (top 18 of 360) of predictions.*

We combine five orthogonal families of signals between the target model and
each suspect:

1. ``weight_detector``       - cosine / scaled-L2 over flattened parameters,
                               computed both globally and on layer groups
                               (early, backbone, head, BatchNorm). Catches
                               near-exact copies.
2. ``behavior_detector``     - logit cosine / agreement / KL / confidence and
                               loss correlation / hard-example similarity /
                               top-5 label-set Jaccard, all on a CIFAR-100
                               test split. Catches behavioural distillation.
3. ``feature_detector``      - average cosine of the penultimate ``avgpool``
                               activations on the test split. Catches stolen
                               models whose internal representation matches
                               the target even when surface predictions differ.
4. ``fingerprint_detector``  - the behavioural signals re-applied to the
                               *target's* training subset (from
                               ``train_main_idx.json``) plus the difference
                               between target-train and non-target-train
                               behaviour (a membership-inference-style gap).
5. ``noise_detector``        - the same behavioural signals on a fixed batch
                               of Gaussian-noise inputs. Independent
                               ResNet-18s give noise-logit cosines in
                               ~0.1-0.3, whereas stolen / distilled copies
                               stay close to 1.0; this is the most powerful
                               single feature.

The final score is the maximum of these family detectors and three composite
detectors (behaviour+feature, behaviour+fingerprint, noise+behaviour) followed
by a final rank-normalisation and a mild sharpening ``s <- s^1.15``. The max
operator is the natural choice for TPR@5%FPR because any single confidently-
firing detector should be sufficient.

Inputs
------
    target_model/weights.safetensors          (required)
    target_model/train_main_idx.json          (optional but strongly recommended;
                                               enables the fingerprint detector)
    suspect_models/<id>/weights.safetensors   or suspect_models/<id>.safetensors
    data/                                     CIFAR-100 (auto-downloaded)

Outputs
-------
    submission.csv      Leaderboard format: two columns (id, score), 360 rows.
    debug_scores.csv    Every raw and ranked column plus every detector.
"""

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


# =============================================================================
# Configuration
# =============================================================================

TARGET_CHECKPOINT = "target_model/weights.safetensors"
SUSPECT_DIR = "suspect_models"
DATA_ROOT = "./data"

# Either of these locations is accepted for the training-index file.
TRAIN_MAIN_IDX_CANDIDATES = (
    "train_main_idx.json",
    "target_model/train_main_idx.json",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_WORKERS = 0

# Probe sizes (number of inputs per family).
N_TEST_PROBE = 5000        # CIFAR-100 test split
N_TRAIN_PROBE = 5000       # subset of the target training indices
N_NONTRAIN_PROBE = 5000    # subset of the remaining CIFAR-100 train images
N_NOISE_PROBE = 512        # fixed Gaussian-noise inputs (OOD probe)

# Behavioural sub-probe sizes.
N_LOW_MARGIN = 300         # hardest target examples for focused similarity
TOPK_JACCARD = 5           # k for top-k label-set Jaccard agreement
NOISE_SEED = 0             # RNG seed for the Gaussian probe (reproducibility)

# CIFAR-100 normalisation statistics (must match the target's training pipeline).
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


# =============================================================================
# Model definition
# =============================================================================

def make_model() -> nn.Module:
    """Return a CIFAR-style ResNet-18 with 100 output classes.

    Differs from the torchvision default in two places to match the target:
        * ``conv1`` is 3x3 stride-1 (the default is 7x7 stride-2 for ImageNet);
        * the initial ``maxpool`` is replaced by ``Identity`` because the input
          is already 32x32.
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


# =============================================================================
# Small numerical / I/O utilities
# =============================================================================

def get_model_id(path: str) -> int:
    """Extract the integer suspect id from a checkpoint path.

    Accepts either ``suspect_models/123.safetensors`` or
    ``suspect_models/123/weights.safetensors`` style layouts.
    """
    p = Path(path)

    m = re.search(r"(\d+)", p.name)
    if m:
        return int(m.group(1))

    for part in reversed(p.parts):
        if part.isdigit():
            return int(part)

    raise ValueError(f"Could not extract model id from {path}")


def rank_normalize(values) -> np.ndarray:
    """Map values to ``[0, 1]`` by rank (higher input -> higher rank).

    Ties are broken by argsort order (stable). Returns ones for a single-element
    input to keep the output well-defined.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.ones_like(values)

    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / (len(values) - 1)


def safe_rank_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Add a ``<col>_rank`` companion column for each name in ``cols``."""
    for col in cols:
        df[col + "_rank"] = rank_normalize(df[col].values)
    return df


def safe_corrcoef(a, b) -> float:
    """Pearson correlation between two 1-D tensors, robust to degenerate inputs."""
    a = torch.as_tensor(a).float().view(-1)
    b = torch.as_tensor(b).float().view(-1)

    if a.numel() < 2 or b.numel() < 2:
        return 0.0

    a = a - a.mean()
    b = b - b.mean()

    denom = a.norm().item() * b.norm().item()
    if denom < 1e-12:
        return 0.0
    return torch.dot(a, b).item() / (denom + 1e-12)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two 1-D tensors. Returns -1.0 on degenerate inputs."""
    denom = a.norm().item() * b.norm().item()
    if denom < 1e-12:
        return -1.0
    return torch.dot(a, b).item() / (denom + 1e-12)


def l2_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Scaled L2 similarity in ``[0, 1]``: 1 - ||a - b|| / (||a|| + ||b||)."""
    dist = torch.norm(a - b).item()
    denom = torch.norm(a).item() + torch.norm(b).item() + 1e-12
    return 1.0 - dist / denom


def flatten_matching_params(sd1, sd2, include_fn=None):
    """Flatten and concatenate parameters that exist (with matching shape) in both
    state dicts.

    Parameters
    ----------
    sd1, sd2 : dict[str, torch.Tensor]
        Source state dicts.
    include_fn : callable or None
        Optional predicate ``key -> bool``. If provided, only keys for which it
        returns True are included.

    Returns
    -------
    (x, y) : tuple of 1-D tensors, or ``(None, None)`` if no parameters match.
    """
    xs, ys = [], []

    for k in sorted(set(sd1.keys()) & set(sd2.keys())):
        if include_fn is not None and not include_fn(k):
            continue

        a, b = sd1[k], sd2[k]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            continue
        if a.shape != b.shape or a.numel() <= 1:
            continue

        xs.append(a.detach().float().cpu().reshape(-1))
        ys.append(b.detach().float().cpu().reshape(-1))

    if not xs:
        return None, None
    return torch.cat(xs), torch.cat(ys)


def weight_group_similarity(target_sd, suspect_sd, include_fn):
    """Return ``(cos, l2_sim)`` for the parameter subset selected by ``include_fn``."""
    x, y = flatten_matching_params(target_sd, suspect_sd, include_fn=include_fn)
    if x is None:
        return -1.0, -1.0
    return cosine_sim(x, y), l2_sim(x, y)


# Named layer-groups used by the weight detector. Each value is a predicate on
# state-dict keys.
WEIGHT_GROUPS = {
    "weight":          lambda k: True,
    "early_weight":    lambda k: k.startswith(("conv1", "bn1", "layer1", "layer2")),
    "backbone_weight": lambda k: k.startswith(("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")),
    "head_weight":     lambda k: k.startswith("fc"),
    "bn_weight":       lambda k: ("running_mean" in k) or ("running_var" in k) or ("bn" in k),
}


def compute_weight_similarities(target_sd, suspect_sd) -> dict:
    """Compute global and grouped (cos, l2) similarities between two state dicts.

    Returns a flat dict with keys ``{group}_cos`` and ``{group}_l2`` for every
    entry in :data:`WEIGHT_GROUPS`.
    """
    out = {}
    for name, fn in WEIGHT_GROUPS.items():
        cos, l2 = weight_group_similarity(target_sd, suspect_sd, include_fn=fn)
        out[f"{name}_cos"] = cos
        out[f"{name}_l2"] = l2
    return out


# =============================================================================
# Training-index parsing (enables the fingerprint detector)
# =============================================================================

def resolve_train_main_idx_path():
    """Return the first existing path from :data:`TRAIN_MAIN_IDX_CANDIDATES`, or
    ``None`` if none of them exists.
    """
    for path in TRAIN_MAIN_IDX_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def load_train_main_indices(train_len: int):
    """Load and validate the target-training indices.

    The JSON may be either a list of integers or a dict containing one (under a
    well-known key or as the first list-valued value).

    Returns a sorted unique list of indices in ``[0, train_len)``, or ``None``
    if no file was found (in which case fingerprint signals will be disabled).
    """
    path = resolve_train_main_idx_path()
    if path is None:
        print(
            "WARNING: train_main_idx.json not found. "
            "Training-fingerprint features will be disabled."
        )
        return None

    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        for key in ("indices", "idx", "train_idx", "train_main_idx", "train_indices"):
            if key in raw:
                raw = raw[key]
                break
        else:
            for value in raw.values():
                if isinstance(value, list):
                    raw = value
                    break
            else:
                raise ValueError(f"Could not parse indices from {path}")

    idx = sorted({int(i) for i in raw if 0 <= int(i) < train_len})
    print(f"Loaded {len(idx)} target training indices from {path}")
    return idx


# =============================================================================
# Data and probe construction
# =============================================================================

def build_datasets():
    """Build the CIFAR-100 train and test datasets with the target-model
    normalisation.

    Downloads to :data:`DATA_ROOT` on first use.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_dataset = datasets.CIFAR100(
        root=DATA_ROOT, train=True, download=True, transform=transform,
    )
    test_dataset = datasets.CIFAR100(
        root=DATA_ROOT, train=False, download=True, transform=transform,
    )
    return train_dataset, test_dataset


def make_loader(dataset, indices, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Return a deterministic ``DataLoader`` over the requested subset."""
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def make_probe_loaders(train_dataset, test_dataset) -> dict:
    """Build all *natural-image* probe loaders the detector uses.

    Always present:
        ``test`` - first :data:`N_TEST_PROBE` CIFAR-100 test images.

    Conditional on a valid ``train_main_idx.json``:
        ``target_train``     - first :data:`N_TRAIN_PROBE` target-training samples.
        ``non_target_train`` - first :data:`N_NONTRAIN_PROBE` remaining CIFAR-100
                                train samples.
    """
    loaders = {}

    test_indices = list(range(min(N_TEST_PROBE, len(test_dataset))))
    loaders["test"] = make_loader(test_dataset, test_indices)

    train_main_idx = load_train_main_indices(len(train_dataset))
    if train_main_idx:
        train_probe_idx = train_main_idx[: min(N_TRAIN_PROBE, len(train_main_idx))]
        loaders["target_train"] = make_loader(train_dataset, train_probe_idx)

        train_main_set = set(train_main_idx)
        nontrain_idx = [i for i in range(len(train_dataset)) if i not in train_main_set]
        nontrain_idx = nontrain_idx[: min(N_NONTRAIN_PROBE, len(nontrain_idx))]
        if nontrain_idx:
            loaders["non_target_train"] = make_loader(train_dataset, nontrain_idx)

    return loaders


def make_noise_probe() -> torch.Tensor:
    """Return a fixed ``(N_NOISE_PROBE, 3, 32, 32)`` Gaussian-noise tensor,
    normalised with CIFAR-100 statistics.

    The same seed is reused for the target and every suspect so the comparison
    is exact. This OOD probe is empirically the most discriminative feature
    in the detector: independent models give noise-logit cosines around
    0.1-0.3, whereas stolen / distilled copies stay close to 1.0.
    """
    g = torch.Generator(device="cpu").manual_seed(NOISE_SEED)
    noise = torch.randn(N_NOISE_PROBE, 3, 32, 32, generator=g)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (noise - mean) / std


# =============================================================================
# Forward-pass collection
# =============================================================================

def collect_outputs(model: nn.Module, loader: DataLoader, collect_features: bool = True) -> dict:
    """Run ``model`` over ``loader`` and gather the quantities we compare on.

    Returns a dict with:
        logits, preds, labels                (always)
        features                             (``collect_features=True`` only;
                                              flattened ``avgpool`` output)
        probs, conf, margin, loss            (derived)
    """
    all_logits, all_preds, all_labels, all_feats = [], [], [], []

    handle = None
    if collect_features:
        def hook_fn(_module, _inputs, output):
            all_feats.append(torch.flatten(output.detach().cpu(), 1))

        handle = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            all_logits.append(logits.detach().cpu())
            all_preds.append(logits.argmax(dim=1).detach().cpu())
            all_labels.append(yb.detach().cpu())

    if handle is not None:
        handle.remove()

    result = {
        "logits": torch.cat(all_logits),
        "preds": torch.cat(all_preds),
        "labels": torch.cat(all_labels),
    }
    if collect_features:
        result["features"] = torch.cat(all_feats)

    probs = F.softmax(result["logits"], dim=1)
    top2 = torch.topk(result["logits"], k=2, dim=1).values
    result["probs"] = probs
    result["conf"] = probs.max(dim=1).values
    result["margin"] = top2[:, 0] - top2[:, 1]
    result["loss"] = F.cross_entropy(result["logits"], result["labels"], reduction="none")
    return result


def compare_outputs(target_out: dict, suspect_out: dict, prefix: str) -> dict:
    """Compute the nine behavioural similarity features for one probe split.

    The returned dict has the keys

        {prefix}_logit_cos              average per-sample cosine of logits
        {prefix}_pred_agree             top-1 prediction agreement
        {prefix}_neg_kl                 - KL(target_probs || suspect_log_probs)
        {prefix}_conf_corr              Pearson corr of max-prob confidence
        {prefix}_loss_corr              Pearson corr of per-sample CE loss
        {prefix}_feature_cos            avgpool feature cosine (or -1 if missing)
        {prefix}_low_margin_logit_cos   logit cosine on N_LOW_MARGIN hardest samples
        {prefix}_low_margin_agree       top-1 agreement on those hardest samples
        {prefix}_topk_jaccard           |top-k intersect| / k (k = TOPK_JACCARD)
    """
    target_logits = target_out["logits"]
    suspect_logits = suspect_out["logits"]
    target_preds = target_out["preds"]
    suspect_preds = suspect_out["preds"]

    suspect_log_probs = F.log_softmax(suspect_logits, dim=1)
    kl = F.kl_div(suspect_log_probs, target_out["probs"], reduction="batchmean").item()

    logit_cos = F.cosine_similarity(target_logits, suspect_logits, dim=1).mean().item()
    pred_agree = (target_preds == suspect_preds).float().mean().item()
    neg_kl = -kl

    conf_corr = safe_corrcoef(target_out["conf"], suspect_out["conf"])
    loss_corr = safe_corrcoef(target_out["loss"], suspect_out["loss"])

    if "features" in target_out and "features" in suspect_out:
        feature_cos = F.cosine_similarity(
            target_out["features"], suspect_out["features"], dim=1,
        ).mean().item()
    else:
        feature_cos = -1.0

    # Focus on the hardest target examples: stolen models tend to retain the
    # target's behaviour on these much better than independently-trained ones.
    low_margin_k = min(N_LOW_MARGIN, len(target_out["margin"]))
    low_margin_idx = torch.argsort(target_out["margin"])[:low_margin_k]
    low_margin_logit_cos = F.cosine_similarity(
        target_logits[low_margin_idx],
        suspect_logits[low_margin_idx],
        dim=1,
    ).mean().item()
    low_margin_agree = (
        target_preds[low_margin_idx] == suspect_preds[low_margin_idx]
    ).float().mean().item()

    # Top-k label-set agreement adds rank information beyond argmax: distilled
    # models often preserve the target's runners-up even if top-1 sometimes
    # differs.
    k = min(TOPK_JACCARD, target_logits.shape[1])
    t_topk = torch.topk(target_logits, k, dim=1).indices
    s_topk = torch.topk(suspect_logits, k, dim=1).indices
    jaccard_vals = torch.tensor(
        [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(t_topk, s_topk)],
        dtype=torch.float32,
    )
    topk_jaccard = (jaccard_vals / k).mean().item()

    return {
        f"{prefix}_logit_cos": logit_cos,
        f"{prefix}_pred_agree": pred_agree,
        f"{prefix}_neg_kl": neg_kl,
        f"{prefix}_conf_corr": conf_corr,
        f"{prefix}_loss_corr": loss_corr,
        f"{prefix}_feature_cos": feature_cos,
        f"{prefix}_low_margin_logit_cos": low_margin_logit_cos,
        f"{prefix}_low_margin_agree": low_margin_agree,
        f"{prefix}_topk_jaccard": topk_jaccard,
    }


def collect_noise_outputs(model: nn.Module, noise_tensor: torch.Tensor) -> dict:
    """Forward the fixed Gaussian-noise tensor and return ``{logits, probs, preds}``."""
    model.eval()
    with torch.no_grad():
        logits = model(noise_tensor.to(DEVICE, non_blocking=True)).cpu()
    return {
        "logits": logits,
        "probs": F.softmax(logits, dim=1),
        "preds": logits.argmax(dim=1),
    }


def compare_noise_outputs(target_noise: dict, suspect_noise: dict) -> dict:
    """Three OOD-probe features: logit cosine, top-1 agreement, negative KL."""
    t_logits = target_noise["logits"]
    s_logits = suspect_noise["logits"]

    noise_logit_cos = F.cosine_similarity(t_logits, s_logits, dim=1).mean().item()
    noise_pred_agree = (target_noise["preds"] == suspect_noise["preds"]).float().mean().item()
    s_log_probs = F.log_softmax(s_logits, dim=1)
    noise_neg_kl = -F.kl_div(s_log_probs, target_noise["probs"], reduction="batchmean").item()

    return {
        "noise_logit_cos": noise_logit_cos,
        "noise_pred_agree": noise_pred_agree,
        "noise_neg_kl": noise_neg_kl,
    }


# =============================================================================
# Detector construction (operates on the per-suspect feature dataframe)
# =============================================================================

def build_fingerprint_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the training-fingerprint detector column on ``df``.

    Requires ``target_train_*`` columns to exist (otherwise the column is set
    to 0). If a ``non_target_train_*`` probe is also present, three additional
    ``train_nontrain_*_gap`` columns and their ranks are added, capturing the
    *difference* between behaviour on the target's training set and on the
    rest of CIFAR-100 train. Stolen models that re-used the target's training
    data show a much larger gap than independent models.
    """
    if "target_train_logit_cos_rank" not in df.columns:
        df["fingerprint_detector"] = 0.0
        return df

    if "non_target_train_logit_cos" in df.columns:
        df["train_nontrain_logit_gap"] = (
            df["target_train_logit_cos"] - df["non_target_train_logit_cos"]
        )
        df["train_nontrain_conf_gap"] = (
            df["target_train_conf_corr"] - df["non_target_train_conf_corr"]
        )
        df["train_nontrain_loss_gap"] = (
            df["target_train_loss_corr"] - df["non_target_train_loss_corr"]
        )
        df = safe_rank_columns(
            df,
            [
                "train_nontrain_logit_gap",
                "train_nontrain_conf_gap",
                "train_nontrain_loss_gap",
            ],
        )
        gap_part = (
            0.12 * df["train_nontrain_logit_gap_rank"]
            + 0.08 * df["train_nontrain_conf_gap_rank"]
            + 0.05 * df["train_nontrain_loss_gap_rank"]
        )
    else:
        gap_part = 0.0

    df["fingerprint_detector"] = (
        0.20 * df["target_train_logit_cos_rank"]
        + 0.15 * df["target_train_neg_kl_rank"]
        + 0.15 * df["target_train_conf_corr_rank"]
        + 0.13 * df["target_train_loss_corr_rank"]
        + 0.13 * df["target_train_feature_cos_rank"]
        + 0.05 * df["target_train_low_margin_logit_cos_rank"]
        + 0.07 * df["target_train_topk_jaccard_rank"]
        + gap_part
    )
    return df


def build_family_detectors(df: pd.DataFrame) -> pd.DataFrame:
    """Build the five family detectors plus the three composite detectors.

    Assumes every raw feature column already has a ``_rank`` companion (i.e.
    :func:`safe_rank_columns` has been called on the full feature dataframe).
    """
    df["weight_detector"] = (
        0.30 * df["weight_cos_rank"]
        + 0.15 * df["weight_l2_rank"]
        + 0.25 * df["backbone_weight_cos_rank"]
        + 0.15 * df["early_weight_cos_rank"]
        + 0.10 * df["bn_weight_cos_rank"]
        + 0.05 * df["head_weight_cos_rank"]
    )

    df["behavior_detector"] = (
        0.20 * df["test_logit_cos_rank"]
        + 0.08 * df["test_pred_agree_rank"]
        + 0.16 * df["test_neg_kl_rank"]
        + 0.13 * df["test_conf_corr_rank"]
        + 0.08 * df["test_loss_corr_rank"]
        + 0.13 * df["test_low_margin_logit_cos_rank"]
        + 0.08 * df["test_low_margin_agree_rank"]
        + 0.14 * df["test_topk_jaccard_rank"]
    )

    df["feature_detector"] = df["test_feature_cos_rank"]

    df["noise_detector"] = (
        0.55 * df["noise_logit_cos_rank"]
        + 0.20 * df["noise_pred_agree_rank"]
        + 0.25 * df["noise_neg_kl_rank"]
    )

    df = build_fingerprint_detector(df)

    # Put each family on a common [0, 1] scale before fusion. Some of the
    # weighted sums above can sum to slightly more than 1.
    for col in ("weight_detector", "behavior_detector", "feature_detector",
                "fingerprint_detector", "noise_detector"):
        df[col] = rank_normalize(df[col].values)

    # Composite detectors mix two complementary signals each.
    #   behavior + feature      : distilled / fine-tuned (behaviour matches,
    #                             weights may not)
    #   behavior + fingerprint  : same but reinforced by training-fingerprint
    #   noise    + behavior     : OOD evidence supported by natural-image agreement
    df["behavior_feature_detector"] = (
        0.65 * df["behavior_detector"] + 0.35 * df["feature_detector"]
    )
    df["behavior_fingerprint_detector"] = (
        0.55 * df["behavior_detector"] + 0.45 * df["fingerprint_detector"]
    )
    df["noise_behavior_detector"] = (
        0.55 * df["noise_detector"] + 0.45 * df["behavior_detector"]
    )
    return df


def compute_final_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the detectors into a final ``score`` column.

    The metric is TPR@5%FPR, so we use ``max`` over the five composite +
    family detectors: any single strong signal should push a true stolen
    model into the top 5% of predictions. The result is then rank-normalised
    (to put it back on a comparable scale across runs) and mildly sharpened
    with ``s <- s^1.15`` to spread out the top of the distribution. We tried
    averaging-style fusion (e.g. trimmed mean) and it consistently
    *underperformed* this max because it diluted single-signal-strong stolens.
    """
    df["score"] = np.maximum.reduce([
        df["weight_detector"].values,
        df["behavior_feature_detector"].values,
        df["behavior_fingerprint_detector"].values,
        df["noise_detector"].values,
        df["noise_behavior_detector"].values,
    ])
    df["score"] = rank_normalize(df["score"].values)
    df["score"] = np.power(df["score"].values, 1.15)
    df["score"] = np.clip(df["score"].values, 0.0, 1.0)
    return df


# =============================================================================
# Per-suspect scoring
# =============================================================================

def score_suspect(
    model_id: int,
    suspect_sd: dict,
    target_sd: dict,
    target_outputs: dict,
    target_noise: dict,
    loaders: dict,
    noise_probe: torch.Tensor,
) -> dict:
    """Compute every raw feature column for a single suspect.

    The returned dict has one entry per raw feature and no rank columns;
    rank-normalisation is applied later in :func:`main` once all 360 suspect
    rows are collected.
    """
    row = {"id": model_id}

    # Weight-only similarities (no forward pass needed).
    row.update(compute_weight_similarities(target_sd, suspect_sd))

    # Behavioural similarities on every natural-image probe split.
    suspect_model = make_model()
    suspect_model.load_state_dict(suspect_sd, strict=True)
    suspect_model = suspect_model.to(DEVICE)
    suspect_model.eval()

    for split_name, loader in loaders.items():
        suspect_out = collect_outputs(suspect_model, loader, collect_features=True)
        row.update(compare_outputs(target_outputs[split_name], suspect_out, split_name))
        del suspect_out

    # Out-of-distribution probe.
    suspect_noise = collect_noise_outputs(suspect_model, noise_probe)
    row.update(compare_noise_outputs(target_noise, suspect_noise))

    del suspect_noise, suspect_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> None:
    print("Starting stolen-model scoring")
    print("Device:", DEVICE)

    # ---- 1. Target model and probes ----
    print("Loading target model...")
    target_sd = load_file(TARGET_CHECKPOINT, device="cpu")
    target_model = make_model()
    target_model.load_state_dict(target_sd, strict=True)
    target_model = target_model.to(DEVICE)
    target_model.eval()

    print("Loading CIFAR-100...")
    train_dataset, test_dataset = build_datasets()
    loaders = make_probe_loaders(train_dataset, test_dataset)
    print("Probe splits:", list(loaders.keys()))

    print("Computing target outputs...")
    target_outputs = {}
    for name, loader in loaders.items():
        print(f"  target split: {name}")
        target_outputs[name] = collect_outputs(target_model, loader, collect_features=True)

    print(f"  target split: gaussian_noise (N={N_NOISE_PROBE})")
    noise_probe = make_noise_probe()
    target_noise = collect_noise_outputs(target_model, noise_probe)

    # ---- 2. Per-suspect scoring ----
    suspect_files = sorted(
        glob.glob(os.path.join(SUSPECT_DIR, "**", "*.safetensors"), recursive=True)
    )
    print("Number of suspect models found:", len(suspect_files))

    rows = []
    for i, suspect_path in enumerate(suspect_files):
        model_id = get_model_id(suspect_path)
        print(f"[{i + 1}/{len(suspect_files)}] scoring suspect id {model_id}")

        suspect_sd = load_file(suspect_path, device="cpu")
        row = score_suspect(
            model_id=model_id,
            suspect_sd=suspect_sd,
            target_sd=target_sd,
            target_outputs=target_outputs,
            target_noise=target_noise,
            loaders=loaders,
            noise_probe=noise_probe,
        )
        rows.append(row)

    # ---- 3. Validate ids ----
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

    if len(df) != 360:
        print("WARNING: expected 360 suspect models, found", len(df))

    missing = sorted(set(range(360)) - set(df["id"].tolist()))
    if missing:
        raise ValueError(f"Missing model ids: {missing[:20]}")
    if df["id"].duplicated().any():
        raise ValueError("Duplicate model ids found")

    # ---- 4. Rank-normalise raw features and build detectors ----
    detector_cols = [c for c in df.columns if c != "id"]
    df = safe_rank_columns(df, detector_cols)
    df = build_family_detectors(df)
    df = compute_final_score(df)

    # ---- 5. Write outputs ----
    submission_df = df[["id", "score"]]
    submission_df.to_csv("submission.csv", index=False)
    df.to_csv("debug_scores.csv", index=False)

    print("Saved submission.csv")
    print("Saved debug_scores.csv")
    print(submission_df.head())
    print(submission_df.tail())


if __name__ == "__main__":
    main()
