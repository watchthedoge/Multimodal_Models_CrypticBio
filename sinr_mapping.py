# sinr_mapping.py
# Maps CrypticBio scientific names -> SINR output indices.
# Two-step lookup: scientific name -> iNaturalist taxon_id -> SINR class index.
# Results cached to disk so the iNat API is hit only once per species set.

import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

INAT_API = "https://api.inaturalist.org/v1/taxa"
CACHE_DIR = Path("sinr_cache")
CACHE_DIR.mkdir(exist_ok=True)
NAME_TO_TAXON_CACHE = CACHE_DIR / "name_to_taxon_id.json"


def _load_cache():
    if NAME_TO_TAXON_CACHE.exists():
        return json.loads(NAME_TO_TAXON_CACHE.read_text())
    return {}


def _save_cache(cache):
    NAME_TO_TAXON_CACHE.write_text(json.dumps(cache, indent=2))


def lookup_inat_taxon_id(scientific_name, cache):
    if scientific_name in cache:
        return cache[scientific_name]

    try:
        r = requests.get(
            INAT_API,
            params={"q": scientific_name, "rank": "species", "per_page": 1},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        # Prefer exact name match; fall back to top hit.
        taxon_id = None
        for hit in results:
            if hit.get("name", "").lower() == scientific_name.lower():
                taxon_id = hit["id"]
                break
        if taxon_id is None and results:
            taxon_id = results[0]["id"]
    except Exception:
        taxon_id = None

    cache[scientific_name] = taxon_id
    return taxon_id


def build_species_index_map(unique_names, sinr_class_to_taxa):
    """
    Returns
    -------
    sinr_idx : list[int]   length 158, with -1 for species not found in SINR
    covered  : list[bool]  True where SINR has a corresponding output
    """
    cache = _load_cache()
    taxa_to_class = {tid: i for i, tid in enumerate(sinr_class_to_taxa)}

    sinr_idx = []
    covered = []
    misses = []
    for name in tqdm(unique_names, desc="Mapping species to SINR"):
        tid = lookup_inat_taxon_id(name, cache)
        # Be polite to the iNat API.
        if tid is None or name not in cache:
            time.sleep(0.4)
        cls_idx = taxa_to_class.get(tid, -1) if tid is not None else -1
        sinr_idx.append(cls_idx)
        covered.append(cls_idx >= 0)
        if cls_idx < 0:
            misses.append(name)

    _save_cache(cache)

    n_total = len(unique_names)
    n_covered = sum(covered)
    print(f"SINR coverage: {n_covered}/{n_total} ({100 * n_covered / n_total:.1f}%)")
    if misses:
        print(f"Missing from SINR ({len(misses)}): {misses[:10]}{'...' if len(misses) > 10 else ''}")

    return sinr_idx, covered
