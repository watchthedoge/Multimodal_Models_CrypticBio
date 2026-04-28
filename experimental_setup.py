from tqdm import tqdm

from dataset_loader import retrieve_dataset
import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import open_clip
from utils import encode_date
from torch.utils.data import DataLoader

BATCH = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup():
    #loading files
    # ds = retrieve_dataset(file="full")
    common = retrieve_dataset(file="common")
    #get unique pairs from the benchmark
    unique_names = sorted(set(common["scientificName"]))
    #batch filtering to get the common subset
    # common_subset = ds.filter(
    #     lambda x: [
    #         name in unique_names 
    #         for name in x['scientificName']
    #     ],
    #     batched=True,
    #     batch_size=10000
    # )


    # Load the model and preprocessors
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

    # Open the image from the URL
    def open_image(url : list[str]):
        try:
            response = requests.get(url, timeout=10)
            return Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
            return None

    #preprocess the image and add batch dimension  
    def preprocess_image(sample: dict):
        img_url = sample['url']
        image = open_image(img_url)
        if image is not None:
            return preprocess_val(image).unsqueeze(0)  #add batch dimension
        else:
            return None

    common = common.filter(
        lambda x: (
            x["month"] is not None
            and x["day"] is not None
            and x["decimalLatitude"] is not None
            and x["decimalLongitude"] is not None
        )
    )
 
    # scientific_names = common_subset["scientificName"]
    # unique_names = sorted(set(scientific_names))
    name_to_idx = {name: i for i, name in enumerate(unique_names)}
 
 
    # Encode every species name with BioCLIP's text encoder
    species_text_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_names), BATCH), desc="Encoding species names"):
            batch = unique_names[i : i + BATCH]
            tokens = tokenizer(batch).to(device)
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
 
    # ---------------------------------------------------- test image collection
 
    test_labels         = []
    test_images         = []
    preprocessed_images = []
    preprocessed_dates  = []
    coords_list         = []
    target, failed      = 1_000, 0
 
    for batch in test_loader:
        if len(preprocessed_images) >= target:
            break
        batch_size = len(batch["url"])
        for i in range(batch_size):
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