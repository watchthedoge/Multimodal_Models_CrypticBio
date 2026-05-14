import re
import time
import math
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_FILE = Path(__file__).parent / "output_1842026.log"
CSV_OUT  = Path(__file__).parent / "full_analysis.csv"
PNG_OUT  = Path(__file__).parent / "full_analysis.png"


RE_CLIP  = re.compile(r"Test Image (\d+): pred='(.+?)'\s+p=(\d+\.\d+)\s+correct=(True|False)")
RE_FUSED = re.compile(r"^Image (\d+): CLIP=(.+?), Fused=(.+?) \(p=(\d+\.\d+)\), True=(.+)$")


def parse_log(path: Path):
    clip_data = {}   # idx → {clip_pred, clip_conf, clip_correct}
    date_data = {}   # idx → {date_pred, date_conf, true_label}
    gps_data  = {}   # idx → {gps_pred,  gps_conf,  true_label}

    # States: before → clip_only → date_fused → gps_training → gps_fused
    state = "before"

    for line in path.read_text().splitlines():
        line = line.strip()

        # State transitions
        if state == "before":
            if RE_CLIP.match(line):
                state = "clip_only"
            # fall through: also parse this first line as content below

        elif state == "clip_only":
            if line.startswith("CLIP-only correct:"):
                state = "date_fused"
                continue  # summary line, not a data row

        elif state == "date_fused":
            if re.match(r"Epoch 0 \| train", line):
                state = "gps_training"
                continue

        elif state == "gps_training":
            if re.match(r"Epoch 9 \| train", line):
                state = "gps_fused"
                continue

        # Content parsing
        if state == "clip_only":
            m = RE_CLIP.match(line)
            if m:
                clip_data[int(m.group(1))] = {
                    "clip_pred":    m.group(2).strip(),
                    "clip_conf":    float(m.group(3)),
                    "clip_correct": m.group(4) == "True",
                }

        elif state == "date_fused":
            m = RE_FUSED.match(line)
            if m:
                date_data[int(m.group(1))] = {
                    "clip_pred":  m.group(2).strip(),
                    "date_pred":  m.group(3).strip(),
                    "date_conf":  float(m.group(4)),
                    "true_label": m.group(5).strip(),
                }

        elif state == "gps_fused":
            m = RE_FUSED.match(line)
            if m:
                gps_data[int(m.group(1))] = {
                    "clip_pred":  m.group(2).strip(),
                    "gps_pred":   m.group(3).strip(),
                    "gps_conf":   float(m.group(4)),
                    "true_label": m.group(5).strip(),
                }

    print(f"Parsed: {len(clip_data)} CLIP-only  |  {len(date_data)} CLIP+date  |  {len(gps_data)} CLIP+GPS")
    return clip_data, date_data, gps_data



# dataframe -------

def _outcome(clip_pred, fused_pred, true_label):
    if clip_pred == fused_pred:
        return "UNCHANGED"
    if clip_pred != true_label and fused_pred == true_label:
        return "IMPROVED"
    if clip_pred == true_label and fused_pred != true_label:
        return "DEGRADED"
    return "BOTH_WRONG"


def build_dataframe(clip_data, date_data, gps_data):
    records = []
    for idx in sorted(set(clip_data) | set(date_data) | set(gps_data)):
        c = clip_data.get(idx, {})
        d = date_data.get(idx, {})
        g = gps_data.get(idx, {})

        true_label  = d.get("true_label") or g.get("true_label") or ""
        clip_pred   = c.get("clip_pred") or d.get("clip_pred") or g.get("clip_pred") or ""
        clip_conf   = c.get("clip_conf", float("nan"))

        date_pred   = d.get("date_pred", clip_pred)
        date_conf   = d.get("date_conf", float("nan"))

        gps_pred    = g.get("gps_pred", clip_pred)
        gps_conf    = g.get("gps_conf", float("nan"))

        date_changed = (date_pred != clip_pred)
        gps_changed  = (gps_pred  != clip_pred)

        # Cross-modal classification
        if not date_changed and not gps_changed:
            cross = "BOTH_UNCHANGED"
        elif gps_changed and not date_changed:
            cross = "ONLY_GPS_CHANGED"
        elif date_changed and not gps_changed:
            cross = "ONLY_DATE_CHANGED"
        elif date_pred == gps_pred:
            cross = "BOTH_CHANGED_AGREE"
        else:
            cross = "BOTH_CHANGED_DISAGREE"

        records.append({
            "img_idx":      idx,
            "true_label":   true_label,
            "clip_pred":    clip_pred,
            "clip_conf":    clip_conf,
            "clip_correct": clip_pred == true_label,
            "date_pred":    date_pred,
            "date_conf":    date_conf,
            "date_correct": date_pred == true_label,
            "date_changed": date_changed,
            "date_outcome": _outcome(clip_pred, date_pred, true_label),
            "gps_pred":     gps_pred,
            "gps_conf":     gps_conf,
            "gps_correct":  gps_pred == true_label,
            "gps_changed":  gps_changed,
            "gps_outcome":  _outcome(clip_pred, gps_pred, true_label),
            "cross_modal":  cross,
        })

    return pd.DataFrame(records).sort_values("img_idx").reset_index(drop=True)



# gbif ----------

_cache = {}   # species_name → {lat, lon, n_coord, months, n_month}


def gbif_fetch(species_name: str) -> dict:
    if species_name in _cache:
        return _cache[species_name]

    # Resolve canonical taxon key
    taxon_key = None
    try:
        r = requests.get(
            "https://api.gbif.org/v1/species/match",
            params={"name": species_name, "strict": "false"},
            timeout=15,
        )
        taxon_key = r.json().get("usageKey")
    except Exception:
        pass

    params = {"hasCoordinate": "true", "limit": 300}
    if taxon_key:
        params["taxonKey"] = taxon_key
    else:
        params["scientificName"] = species_name

    results = []
    try:
        time.sleep(0.12)
        r = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params=params,
            timeout=20,
        )
        results = r.json().get("results", [])
    except Exception:
        pass

    lats   = [x["decimalLatitude"]  for x in results if x.get("decimalLatitude")  is not None]
    lons   = [x["decimalLongitude"] for x in results if x.get("decimalLongitude") is not None]
    months = [x["month"]            for x in results if x.get("month")            is not None]

    data = {
        "lat":     float(np.median(lats))   if lats   else float("nan"),
        "lon":     float(np.median(lons))   if lons   else float("nan"),
        "n_coord": len(lats),
        "months":  months,
        "n_month": len(months),
    }
    _cache[species_name] = data
    return data


# distance functions -------

def haversine(lat1, lon1, lat2, lon2) -> float:
    if any(math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def circular_peak_month(months: list) -> float:
    """Circular mean month (1–12) from a list of integer months."""
    if len(months) < 3:
        return float("nan")
    angles   = [2 * math.pi * (m - 1) / 12 for m in months]
    mean_sin = sum(math.sin(a) for a in angles) / len(angles)
    mean_cos = sum(math.cos(a) for a in angles) / len(angles)
    mean_ang = math.atan2(mean_sin, mean_cos)
    return (mean_ang * 12 / (2 * math.pi)) % 12 + 1


def pheno_distance_days(peak1: float, peak2: float) -> float:
    """Circular distance between two peak months in days (max ≈ 182)."""
    if math.isnan(peak1) or math.isnan(peak2):
        return float("nan")
    diff = abs(peak1 - peak2)
    return min(diff, 12 - diff) * 30.44


# dataframe with geo data --------------

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    # Collect all species we need to query
    gps_species  = set()
    date_species = set()
    for _, row in df.iterrows():
        if row["gps_changed"]:
            gps_species.update([row["clip_pred"], row["gps_pred"]])
        if row["date_changed"]:
            date_species.update([row["clip_pred"], row["date_pred"]])

    all_species = gps_species | date_species
    if not all_species:
        print("No changed predictions — skipping GBIF queries.")
        df["gps_dist_km"]       = float("nan")
        df["date_pheno_days"]   = float("nan")
        df["clip_peak_month"]   = float("nan")
        df["date_peak_month"]   = float("nan")
        df["gps_low_coverage"]  = False
        df["date_low_coverage"] = False
        return df

    print(f"\nQuerying GBIF for {len(all_species)} unique species …")
    for i, sp in enumerate(sorted(all_species), 1):
        gbif_fetch(sp)
        if i % 10 == 0 or i == len(all_species):
            print(f"  {i}/{len(all_species)} done", end="\r")
    print()

    gps_dist, date_dist   = [], []
    clip_peak, date_peak  = [], []
    gps_low, date_low     = [], []

    for _, row in df.iterrows():
        if row["gps_changed"]:
            ca = gbif_fetch(row["clip_pred"])
            ga = gbif_fetch(row["gps_pred"])
            gps_dist.append(haversine(ca["lat"], ca["lon"], ga["lat"], ga["lon"]))
            gps_low.append(ca["n_coord"] < 10 or ga["n_coord"] < 10)
        else:
            gps_dist.append(float("nan"))
            gps_low.append(False)

        if row["date_changed"]:
            ca = gbif_fetch(row["clip_pred"])
            da = gbif_fetch(row["date_pred"])
            p1 = circular_peak_month(ca["months"])
            p2 = circular_peak_month(da["months"])
            date_dist.append(pheno_distance_days(p1, p2))
            clip_peak.append(p1)
            date_peak.append(p2)
            date_low.append(ca["n_month"] < 10 or da["n_month"] < 10)
        else:
            date_dist.append(float("nan"))
            clip_peak.append(float("nan"))
            date_peak.append(float("nan"))
            date_low.append(False)

    df = df.copy()
    df["gps_dist_km"]       = gps_dist
    df["date_pheno_days"]   = date_dist
    df["clip_peak_month"]   = clip_peak
    df["date_peak_month"]   = date_peak
    df["gps_low_coverage"]  = gps_low
    df["date_low_coverage"] = date_low
    return df


# summary ------

def print_summary(df: pd.DataFrame):
    n = len(df)
    clip_acc = df["clip_correct"].mean()
    date_acc = df["date_correct"].mean()
    gps_acc  = df["gps_correct"].mean()

    print("\n" + "=" * 65)
    print("FULL THREE-WAY ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"Total images: {n}")
    print(f"\nAccuracy:  CLIP={clip_acc:.1%}  |  +date={date_acc:.1%}  |  +GPS={gps_acc:.1%}")
    print(f"  Date net change: {(date_acc - clip_acc)*100:+.1f} pp")
    print(f"  GPS  net change: {(gps_acc  - clip_acc)*100:+.1f} pp")

    for label, col_changed, col_outcome in [
        ("Date", "date_changed", "date_outcome"),
        ("GPS",  "gps_changed",  "gps_outcome"),
    ]:
        sub = df[df[col_changed]]
        vc  = sub[col_outcome].value_counts()
        print(f"\n{label} changes: {len(sub)} / {n}  ({len(sub)/n:.1%})")
        for o in ["IMPROVED", "DEGRADED", "BOTH_WRONG"]:
            cnt = vc.get(o, 0)
            print(f"  {o:<12}: {cnt:>3}  ({cnt/n:.1%} of all images)")

    if "date_pheno_days" in df.columns:
        print("\nMedian phenological distance between species (days):")
        for o in ["IMPROVED", "DEGRADED", "BOTH_WRONG"]:
            sub = df[(df["date_outcome"] == o) & df["date_pheno_days"].notna()]
            if len(sub):
                print(f"  Date {o:<12}: {sub['date_pheno_days'].median():.1f} days  (n={len(sub)})")

    if "gps_dist_km" in df.columns:
        print("\nMedian geographic distance between species centroids (km):")
        for o in ["IMPROVED", "DEGRADED", "BOTH_WRONG"]:
            sub = df[(df["gps_outcome"] == o) & df["gps_dist_km"].notna()]
            if len(sub):
                print(f"  GPS  {o:<12}: {sub['gps_dist_km'].median():.0f} km  (n={len(sub)})")

    print("\nCross-modal interaction:")
    for k, v in df["cross_modal"].value_counts().items():
        sub = df[df["cross_modal"] == k]
        gps_a  = sub["gps_correct"].mean()
        date_a = sub["date_correct"].mean()
        print(f"  {k:<28}: {v:>4}  (GPS acc={gps_a:.0%}  date acc={date_a:.0%})")

    agree = df[df["cross_modal"] == "BOTH_CHANGED_AGREE"]
    if len(agree):
        print(f"\nWhen GPS+date BOTH changed to the SAME species ({len(agree)} cases):")
        print(f"  CLIP acc:  {agree['clip_correct'].mean():.1%}")
        print(f"  GPS  acc:  {agree['gps_correct'].mean():.1%}")
        print(f"  Date acc:  {agree['date_correct'].mean():.1%}")


# printed tables ------

def _show(df_sub, cols, n=10, title=""):
    if title:
        print(f"\n{'─'*75}")
        print(f"  {title}")
        print(f"{'─'*75}")
    cols = [c for c in cols if c in df_sub.columns]
    with pd.option_context("display.max_colwidth", 28, "display.width", 135,
                           "display.float_format", "{:.1f}".format):
        print(df_sub[cols].head(n).to_string(index=False))
    if len(df_sub) == 0:
        print("  (no cases)")


def print_tables(df: pd.DataFrame):
    # 1. Both GPS and date improved the same image
    both_imp = df[(df["gps_outcome"] == "IMPROVED") & (df["date_outcome"] == "IMPROVED")]
    _show(both_imp,
          ["img_idx", "clip_pred", "date_pred", "gps_pred", "true_label",
           "date_pheno_days", "gps_dist_km", "date_conf", "gps_conf"],
          title="BOTH GPS+date IMPROVED — strongest consensus signal")

    # 2. Metadata conflict
    conflict = df[
        ((df["gps_outcome"] == "IMPROVED") & (df["date_outcome"] == "DEGRADED")) |
        ((df["gps_outcome"] == "DEGRADED") & (df["date_outcome"] == "IMPROVED"))
    ].sort_values("img_idx")
    _show(conflict,
          ["img_idx", "clip_pred", "date_pred", "gps_pred", "true_label",
           "date_outcome", "gps_outcome", "date_pheno_days", "gps_dist_km"],
          title="METADATA CONFLICT — GPS and date correct in opposite directions")

    # 3. Both changed but disagreed on species
    disagree = df[df["cross_modal"] == "BOTH_CHANGED_DISAGREE"].sort_values(
        "gps_dist_km", ascending=False, na_position="last"
    )
    _show(disagree,
          ["img_idx", "clip_pred", "date_pred", "gps_pred", "true_label",
           "cross_modal", "gps_dist_km", "date_pheno_days"],
          title="GPS+date BOTH CHANGED but DISAGREED on which species")

    # 4. GPS top improvements by geographic distance
    gps_imp = df[df["gps_outcome"] == "IMPROVED"].sort_values(
        "gps_dist_km", ascending=False, na_position="last"
    )
    _show(gps_imp,
          ["img_idx", "clip_pred", "gps_pred", "true_label", "gps_dist_km", "gps_conf"],
          title="GPS IMPROVEMENTS — sorted by geographic distance")

    # 5. Date top improvements by phenological distance
    date_imp = df[df["date_outcome"] == "IMPROVED"].sort_values(
        "date_pheno_days", ascending=False, na_position="last"
    )
    _show(date_imp,
          ["img_idx", "clip_pred", "date_pred", "true_label",
           "date_pheno_days", "clip_peak_month", "date_peak_month", "date_conf"],
          title="DATE IMPROVEMENTS — sorted by phenological distance (days)")

    # 6. GPS degradations — where did GPS go wrong?
    gps_deg = df[df["gps_outcome"] == "DEGRADED"].sort_values(
        "gps_dist_km", ascending=False, na_position="last"
    )
    _show(gps_deg,
          ["img_idx", "clip_pred", "gps_pred", "true_label", "gps_dist_km", "gps_conf"],
          title="GPS DEGRADATIONS — GPS broke a correct CLIP prediction")


# create figures ------

COLORS = {"IMPROVED": "#2ecc71", "DEGRADED": "#e74c3c", "BOTH_WRONG": "#95a5a6", "UNCHANGED": "#bdc3c7"}
OUTCOMES = ["IMPROVED", "DEGRADED", "BOTH_WRONG"]


def make_plots(df: pd.DataFrame):
    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    fig.suptitle("CLIP vs CLIP+Date vs CLIP+GPS — Full Modality Analysis", fontsize=14, fontweight="bold")

    # 1. Accuracy bar chart
    accs  = [df["clip_correct"].mean(), df["date_correct"].mean(), df["gps_correct"].mean()]
    bars  = ax1.bar(["CLIP only", "+date", "+GPS"], [a * 100 for a in accs],
                    color=["#3498db", "#f39c12", "#2ecc71"], width=0.5, alpha=0.85)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f"{acc:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax1.axhline(accs[0] * 100, color="grey", linestyle=":", linewidth=1, label="CLIP baseline")
    ax1.set_ylim(0, 80)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Classification accuracy by modality")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8)

    # 2. GPS geographic distance boxplot
    data_gps = [df[df["gps_outcome"] == o]["gps_dist_km"].dropna().values for o in OUTCOMES]
    bp2 = ax2.boxplot(data_gps, patch_artist=True, tick_labels=OUTCOMES, widths=0.5)
    for patch, o in zip(bp2["boxes"], OUTCOMES):
        patch.set_facecolor(COLORS[o]); patch.set_alpha(0.75)
    for i, (d, o) in enumerate(zip(data_gps, OUTCOMES), 1):
        if len(d):
            ax2.text(i, np.median(d) + 80, f"{np.median(d):.0f} km",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Geographic distance between species centroids (km)")
    ax2.set_title("GPS: species range distance by outcome")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # 3. Date phenological distance boxplot
    data_date = [df[df["date_outcome"] == o]["date_pheno_days"].dropna().values for o in OUTCOMES]
    bp3 = ax3.boxplot(data_date, patch_artist=True, tick_labels=OUTCOMES, widths=0.5)
    for patch, o in zip(bp3["boxes"], OUTCOMES):
        patch.set_facecolor(COLORS[o]); patch.set_alpha(0.75)
    for i, (d, o) in enumerate(zip(data_date, OUTCOMES), 1):
        if len(d):
            ax3.text(i, np.median(d) + 1, f"{np.median(d):.0f} d",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Phenological distance between species peak seasons (days)")
    ax3.set_title("Date: seasonal separation by outcome")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    # 4. Cross-modal heatmap
    all_out = ["UNCHANGED", "IMPROVED", "DEGRADED", "BOTH_WRONG"]
    matrix  = np.zeros((4, 4), dtype=int)
    for i, do in enumerate(all_out):
        for j, go in enumerate(all_out):
            matrix[i, j] = int(((df["date_outcome"] == do) & (df["gps_outcome"] == go)).sum())

    im = ax4.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax4.set_xticks(range(4)); ax4.set_xticklabels(all_out, rotation=20, ha="right", fontsize=8)
    ax4.set_yticks(range(4)); ax4.set_yticklabels(all_out, fontsize=8)
    ax4.set_xlabel("GPS outcome"); ax4.set_ylabel("Date outcome")
    ax4.set_title("Cross-modal outcome matrix (rows=date, cols=GPS)")
    thresh = matrix.max() * 0.55
    for i in range(4):
        for j in range(4):
            if matrix[i, j] > 0:
                ax4.text(j, i, str(matrix[i, j]), ha="center", va="center",
                         fontsize=10, color="white" if matrix[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax4, shrink=0.8)

    plt.savefig(PNG_OUT, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {PNG_OUT}")


# main -----

def main():
    clip_data, date_data, gps_data = parse_log(LOG_FILE)
    df = build_dataframe(clip_data, date_data, gps_data)
    df = enrich(df)
    print_summary(df)
    print_tables(df)
    make_plots(df)

    save_cols = [
        "img_idx", "true_label",
        "clip_pred",  "clip_conf",  "clip_correct",
        "date_pred",  "date_conf",  "date_correct",  "date_changed",  "date_outcome",
        "gps_pred",   "gps_conf",   "gps_correct",   "gps_changed",   "gps_outcome",
        "cross_modal",
        "gps_dist_km", "date_pheno_days", "clip_peak_month", "date_peak_month",
        "gps_low_coverage", "date_low_coverage",
    ]
    df[save_cols].to_csv(CSV_OUT, index=False, float_format="%.4f")
    print(f"CSV saved  → {CSV_OUT}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
