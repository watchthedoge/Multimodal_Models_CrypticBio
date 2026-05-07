from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from dataset_loader import retrieve_dataset
import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import open_clip
from utils import encode_date
from torch.utils.data import DataLoader
import hashlib


BATCH = 256
CACHE_DIR = Path("image_cache")
CACHE_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _url_to_filename(url: str) -> Path:
    unique_id = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{unique_id}.jpg"


def open_image(url: str, retries: int = 3) -> "Optional[Image.Image]":
    """Return a PIL image, using a local cache to avoid re-downloading.
    Retries on 429 with exponential backoff."""
    import time

    cache_path = _url_to_filename(url)
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            cache_path.unlink(missing_ok=True)  # corrupt cache file — delete and re-fetch

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 429:
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)
                continue
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(cache_path)
            return img
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"Error downloading image: {e}")
    return None


def _prefetch_urls(urls: list[str], max_workers: int = 4) -> None:
    """Download and cache a list of URLs in parallel (best-effort).
    Keep workers low (4) to avoid rate limiting."""
    import time

    todo = [u for u in urls if not _url_to_filename(u).exists()]
    if not todo:
        print("All images already cached.")
        return

    print(f"Pre-fetching {len(todo)} images with {max_workers} workers…")

    def _fetch(url):
        open_image(url)
        time.sleep(0.05)  # small delay per worker to ease rate limiting

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch, u): u for u in todo}
        for _ in tqdm(as_completed(futures), total=len(todo), desc="Downloading"):
            pass


def setup():
    # loading files
    common = retrieve_dataset(file="common")
    unique_names = sorted(set(common["scientificName"]))

    # Load the model and preprocessors
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

    def preprocess_image(sample: dict) -> "Optional[torch.Tensor]":
        img = open_image(sample["url"])
        if img is not None:
            return preprocess_val(img).unsqueeze(0)
        return None

    common = common.filter(
        lambda x: (
            x["month"] is not None
            and x["day"] is not None
            and x["decimalLatitude"] is not None
            and x["decimalLongitude"] is not None
        )
    )

    name_to_idx = {name: i for i, name in enumerate(unique_names)}

    # Encode every species name with BioCLIP's text encoder
    species_text_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_names), BATCH), desc="Encoding species names"):
            batch = unique_names[i : i + BATCH]
            tokens = tokenizer(batch).to(device).long()
            embs = model.encode_text(tokens)
            embs /= embs.norm(dim=-1, keepdim=True)
            species_text_embs.append(embs.cpu())

    species_text_embs = torch.cat(species_text_embs, dim=0)

    processed = common.map(
        lambda x: {
            "encoded_input": encode_date(x["month"], x["day"]),
            "label_idx": name_to_idx[x["scientificName"]],
            "coords": torch.tensor(
                [x["decimalLatitude"], x["decimalLongitude"]], dtype=torch.float32
            ),
        }
    )

    processed.set_format(type="torch", columns=["encoded_input", "label_idx", "url", "coords"])

    split = processed.train_test_split(test_size=0.21, seed=42)
    train_ds, test_ds = split["train"], split["test"]

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=2048, shuffle=False, num_workers=2, pin_memory=True)

    # Pre-compute log-prior over species from the full processed set
    label_array = np.array(processed["label_idx"], dtype=np.int64)
    counts = torch.bincount(torch.from_numpy(label_array), minlength=len(name_to_idx)).float()
    log_prior = torch.log(counts / counts.sum() + 1e-12).to(device)

    target = 1000
    all_test_urls = []
    for batch in test_loader:
        all_test_urls.extend(batch["url"])
        if len(all_test_urls) >= target:
            break
    all_test_urls = all_test_urls[:target]
    _prefetch_urls(all_test_urls)

    # ---- build test collections from cache ----
    test_labels         = []
    test_images         = []
    preprocessed_images = []
    preprocessed_dates  = []
    coords_list         = []
    failed              = 0

    for batch in test_loader:
        if len(preprocessed_images) >= target:
            break
        for i in range(len(batch["url"])):
            if len(preprocessed_images) >= target:
                break
            url = batch["url"][i]
            try:
                img = preprocess_image({"url": url})
                if img is None:
                    failed += 1
                    continue
                preprocessed_images.append(img)
                test_images.append(url)
                test_labels.append(batch["label_idx"][i])
                preprocessed_dates.append(batch["encoded_input"][i])
                coords_list.append(batch["coords"][i])
            except Exception as e:
                failed += 1
                print(f"Skipping {url}: {type(e).__name__}: {e}")

    print(f"Collected {len(preprocessed_images)} test images, failed {failed}")
    assert len(preprocessed_images) == len(test_labels) == len(preprocessed_dates) == len(test_images)

    return {
        "model":               model,
        "tokenizer":           tokenizer,
        "preprocess_image":    preprocess_image,
        "common_subset":       common,
        "processed":           processed,
        "train_loader":        train_loader,
        "test_loader":         test_loader,
        "unique_names":        unique_names,
        "name_to_idx":         name_to_idx,
        "species_text_embs":   species_text_embs,
        "log_prior":           log_prior,
        "preprocessed_images": preprocessed_images,
        "preprocessed_dates":  preprocessed_dates,
        "coords":              coords_list,
        "test_labels":         test_labels,
        "test_images":         test_images,
        "device":              device,
    }