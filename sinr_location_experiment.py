# sinr_location_experiment.py
# Experiment: fuse frozen BioCLIP-2 image probabilities with SINR's
# pretrained geographic prior, in log-space.
#
# FUSION MATH
# -----------
# Full Bayesian factoring:
#     log P(class | image, loc) = log P(class|image) + log P(loc|class)
#                                  - log P(class) + C
#
# In code:
#     fused = clip_log_softmax + alpha * (sinr_log_prior - sinr_background) - log_prior
#     pred  = log_softmax(fused)
#
# CALIBRATION (the key fix vs. naive log(sigmoid) fusion)
# -------------------------------------------------------
# Naive zero-shot SINR fusion fails because SINR's sigmoid outputs encode
# both "is this species likely HERE" and "is this species globally common".
# A globally common species gets a higher absolute sigmoid value almost
# everywhere than a globally rarer species, even at equally suitable
# locations. That global-frequency bias leaks into the argmax.
#
# Fix: per-species background calibration. For each species, compute its
# MEAN log-prior across a sample of random global locations -- the
# "expected log-presence anywhere." Subtract this from the per-sample
# log-prior. The result is a location-relative score:
#   - >0  : species is more present here than typical
#   - <0  : species is less present here than typical
#   - ~0  : indifferent
# This removes the global-frequency offset cleanly.
#
# IMPORT STRATEGY
# ---------------
# Load SINR's modules via importlib from explicit paths to avoid
# datasets.py / utils.py shadowing project modules.

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from experimental_setup import setup
from sinr_mapping import build_species_index_map


# --- Load SINR modules ------------------------------------------------------

SINR_DIR = Path(__file__).parent / "external" / "sinr"


def _load_module_from_file(unique_name, file_path):
    spec = importlib.util.spec_from_file_location(unique_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


sinr_utils = _load_module_from_file("sinr_utils", SINR_DIR / "utils.py")
CoordEncoder = sinr_utils.CoordEncoder

_saved_utils = sys.modules.get("utils")
sys.modules["utils"] = sinr_utils
try:
    sinr_models = _load_module_from_file("sinr_models", SINR_DIR / "models.py")
finally:
    if _saved_utils is not None:
        sys.modules["utils"] = _saved_utils
    else:
        sys.modules.pop("utils", None)


SINR_CHECKPOINT = SINR_DIR / "pretrained_models" / "model_an_full_input_enc_sin_cos_distilled_from_env.pt"


def load_sinr(device):
    ckpt = torch.load(SINR_CHECKPOINT, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    model = sinr_models.get_model(params)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device).eval()
    encoder = CoordEncoder(input_enc=params["input_enc"])
    class_to_taxa = params["class_to_taxa"]
    return model, encoder, class_to_taxa


# --- Per-species background calibration -------------------------------------

def compute_sinr_background(sinr_model, coord_encoder, sinr_idx_t, covered_t,
                            device, n_samples=10000, seed=42):
    """
    For each of our 158 species, compute the mean log(sigmoid(sinr)) across
    n_samples random global locations. This is the species' "expected
    log-presence anywhere," used as a per-species offset to remove the
    global frequency bias from SINR's outputs.

    Sampling uniform over latitude band [-60, 75] (avoids polar caps where
    SINR has essentially no signal) and longitude [-180, 180]. Land/sea
    bias is acceptable for calibration -- we're estimating a per-species
    baseline, not building a presence map.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = (torch.rand(n_samples, generator=g) * 135.0) - 60.0       # [-60, 75]
    lon = (torch.rand(n_samples, generator=g) * 360.0) - 180.0      # [-180, 180]
    lon_lat = torch.stack([lon, lat], dim=-1).float()               # SINR order

    safe_idx = sinr_idx_t.clamp(min=0)
    n_classes = len(sinr_idx_t)

    accumulator = torch.zeros(n_classes, device=device)
    chunk = 1024
    n_done = 0

    with torch.no_grad():
        for start in tqdm(range(0, n_samples, chunk), desc="Calibrating SINR background"):
            batch_lon_lat = lon_lat[start:start + chunk].to(device)
            loc_feats = coord_encoder.encode(batch_lon_lat).to(device)
            logits_all = sinr_model(loc_feats).float()              # (B, ~47k)
            probs_all = torch.sigmoid(logits_all)
            gathered = probs_all.index_select(1, safe_idx).clamp(min=1e-8)  # (B, 158)
            log_gathered = torch.log(gathered)
            accumulator += log_gathered.sum(dim=0)
            n_done += batch_lon_lat.shape[0]

    background = accumulator / n_done                               # (158,)

    # Uncovered species: set background to 0 so calibrated values also
    # end up 0 (neutral -- doesn't favor or penalize them).
    background = torch.where(covered_t, background, torch.zeros_like(background))
    return background


# --- Precompute per-sample log-probability vectors --------------------------

def _precompute_log_probs(ctx, sinr_model, coord_encoder, sinr_idx_t, covered_t,
                          background, device):
    species_text_embs = ctx["species_text_embs"].to(device)
    n_classes = len(ctx["unique_names"])
    n_samples = len(ctx["test_labels"])

    clip_log_probs = torch.empty(n_samples, n_classes, device=device)
    sinr_log_cal = torch.empty(n_samples, n_classes, device=device)

    safe_idx = sinr_idx_t.clamp(min=0)
    zero_fill = torch.zeros(n_classes, device=device)

    with torch.no_grad(), torch.amp.autocast(device.type):
        for i, (img, coord) in enumerate(tqdm(
            list(zip(ctx["preprocessed_images"], ctx["coords"])),
            desc="Precomputing CLIP + SINR (calibrated)",
        )):
            img = img.to(device)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            coord = coord.to(device)

            # CLIP branch
            img_feats = ctx["model"].encode_image(img)
            img_feats = F.normalize(img_feats, dim=-1)
            logit_scale = ctx["model"].logit_scale.exp()
            clip_logits = logit_scale * (img_feats @ species_text_embs.T)
            clip_log_probs[i] = F.log_softmax(clip_logits, dim=-1).squeeze(0).float()

            # SINR branch -- expects [lon, lat], coords stored as [lat, lon]
            lon_lat = torch.stack([coord[1], coord[0]]).unsqueeze(0).float()
            loc_feats = coord_encoder.encode(lon_lat).to(device)
            sinr_logits_all = sinr_model(loc_feats).squeeze(0).float()
            sinr_probs_all = torch.sigmoid(sinr_logits_all)

            gathered = sinr_probs_all.index_select(0, safe_idx).clamp(min=1e-8)
            log_gathered = torch.log(gathered)
            calibrated = log_gathered - background
            calibrated = torch.where(covered_t, calibrated, zero_fill)
            sinr_log_cal[i] = calibrated

    return clip_log_probs, sinr_log_cal


# --- Fusion + scoring -------------------------------------------------------

def _score(clip_log_probs, sinr_log_cal, log_prior, alpha, test_labels, covered):
    """
    Full Bayesian fusion:
        fused = clip_log_probs + alpha * sinr_log_calibrated - log_prior
    Then log_softmax to normalize.
    """
    fused = clip_log_probs + alpha * sinr_log_cal - log_prior
    fused = F.log_softmax(fused, dim=-1)
    preds = fused.argmax(dim=-1).cpu().tolist()

    total = len(test_labels)
    correct_full = sum(int(p == y) for p, y in zip(preds, test_labels))
    covered_pairs = [(p, y) for p, y in zip(preds, test_labels) if covered[y]]
    correct_covered = sum(int(p == y) for p, y in covered_pairs)
    n_covered = len(covered_pairs)

    return {
        "alpha": alpha,
        "acc_full": correct_full / total,
        "acc_covered": correct_covered / max(n_covered, 1),
        "n_total": total,
        "n_covered": n_covered,
    }


# --- Public entry points ----------------------------------------------------

def run(alpha=1.0):
    return run_sweep(alphas=[alpha])


def run_sweep(alphas=(0.0, 5.0, 10.0, 20.0, 25.0, 30.0, 50.0),
              n_background=10000):
    """
    Free alpha sweep with calibrated SINR. alpha=0 reproduces CLIP-only.
    """
    ctx = setup()
    device = ctx["device"]
    log_prior = ctx["log_prior"].to(device)

    print("Loading SINR checkpoint...")
    sinr_model, coord_encoder, class_to_taxa = load_sinr(device)

    print("Building CrypticBio -> SINR species index map...")
    sinr_idx, covered = build_species_index_map(ctx["unique_names"], class_to_taxa)
    sinr_idx_t = torch.tensor(sinr_idx, device=device)
    covered_t = torch.tensor(covered, device=device)

    print(f"Computing per-species SINR background ({n_background} random locations)...")
    background = compute_sinr_background(
        sinr_model, coord_encoder, sinr_idx_t, covered_t, device,
        n_samples=n_background,
    )
    bg_covered = background[covered_t]
    print(f"  background log-prior range (covered species): "
          f"min={bg_covered.min().item():.3f}, "
          f"mean={bg_covered.mean().item():.3f}, "
          f"max={bg_covered.max().item():.3f}")

    clip_log_probs, sinr_log_cal = _precompute_log_probs(
        ctx, sinr_model, coord_encoder, sinr_idx_t, covered_t, background, device,
    )

    print()
    print(f"{'alpha':>8} | {'acc (all 158)':>14} | {'acc (covered 153)':>18}")
    print("-" * 50)
    results = []
    for a in alphas:
        r = _score(clip_log_probs, sinr_log_cal, log_prior, a,
                   ctx["test_labels"], covered)
        results.append(r)
        print(f"{a:>8.2f} | {r['acc_full']:>14.4f} | {r['acc_covered']:>18.4f}")
    print()

    best = max(results, key=lambda r: r["acc_full"])
    print(f"Best alpha = {best['alpha']:.2f}  (acc={best['acc_full']:.4f})")
    return results
