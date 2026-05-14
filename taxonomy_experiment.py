import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from experimental_setup import setup

TAXONOMY_LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "scientificName"]

BATCH = 256

class DateToTaxon(nn.Module):
    def __init__(self, date_dim=4, output_dim=158):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(date_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.network(torch.flatten(x, start_dim=1))


class LocToTaxon(nn.Module):
    def __init__(self, output_dim=158):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(torch.flatten(x, start_dim=1))


def run_experiment(experiment, level):
    ctx = setup()

    device              = ctx["device"]
    model               = ctx["model"]
    tokenizer           = ctx["tokenizer"]
    common_subset       = ctx["common_subset"]
    processed           = ctx["processed"]
    preprocessed_images = ctx["preprocessed_images"]
    preprocessed_dates  = ctx["preprocessed_dates"]
    coords              = ctx["coords"]
    test_images         = ctx["test_images"]

    # Build label space
    unique_labels = sorted(set(common_subset[level]), key=lambda x: (x is None, x))    
    label_to_idx  = {lbl: i for i, lbl in enumerate(unique_labels)}
    print(f"None in label_to_idx: {None in label_to_idx}")
    print(f"Sample values: {list(label_to_idx.keys())[:10]}")
    print(f"\nLevel: {level!r} — {len(unique_labels)} unique labels")

    # One reptile species in CrypticBio-Common has no `order` value. Squamata is its correct order
    if level == "order" and None in label_to_idx:
        idx = label_to_idx.pop(None)
        label_to_idx["Squamata"] = idx
        unique_labels[idx] = "Squamata"

    # BioCLIP only has species-level text embeddings. We use scatter_add_ to aggregate them to the target taxon level
    species_text_embs = ctx["species_text_embs"].to(device)
    unique_species    = ctx["unique_names"]

    # find which taxon index it belongs to at the chosen level
    species_taxon_map = {row["scientificName"]: label_to_idx["Squamata" if row[level] is None else row[level]] for row in common_subset}
    species_to_taxon  = torch.tensor(
        [species_taxon_map[sp] for sp in unique_species], dtype=torch.long
    ).to(device)

    taxon_indices = [label_to_idx["Squamata" if lbl is None else lbl] for lbl in common_subset[level]]
    counts    = torch.bincount(torch.tensor(taxon_indices, dtype=torch.long), minlength=len(label_to_idx)).float()
    log_prior = torch.log(counts / counts.sum() + 1e-12).to(device)

    # Remap to the taxonomy level
    url_to_taxon = {common_subset[i]["url"]: label_to_idx["Squamata" if common_subset[level][i] is None else common_subset[level][i]] for i in range(len(common_subset))}
    remapped = processed.map(lambda x: {"label_idx": url_to_taxon[x["url"]]})
    remapped.set_format(type="torch", columns=["encoded_input", "label_idx", "coords"])
    train_loader = DataLoader(remapped, batch_size=2048, shuffle=True, num_workers=2, pin_memory=True)

    test_labels = [url_to_taxon[url] for url in test_images]

    # Run the chosen experiment(s)
    if experiment in ("date", "both"):
        network   = DateToTaxon(output_dim=len(unique_labels)).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        loss_fn   = nn.CrossEntropyLoss()

        for epoch in range(10):
            total_loss, steps = 0.0, 0
            for batch in train_loader:
                x = batch["encoded_input"].to(device, non_blocking=True)
                y = batch["label_idx"].to(device, non_blocking=True)
                optimizer.zero_grad()
                l = loss_fn(network(x), y)
                l.backward()
                optimizer.step()
                total_loss += l.item(); steps += 1
            print(f"Epoch {epoch} | loss: {total_loss/steps:.4f}")

        network.eval()
        clip_correct, fused_correct, fused_confidences = 0, 0, []

        for i, img in enumerate(preprocessed_images):
            img = img.to(device)
            true_label = test_labels[i]

            with torch.no_grad(), torch.amp.autocast(device.type):
                image_features = model.encode_image(img)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                #score at species level, then sum probs per taxon group
                species_probs = F.softmax(100 * image_features @ species_text_embs.T, dim=-1).float()
                clip_probs    = torch.zeros(1, len(unique_labels), device=device).scatter_add_(
                    1, species_to_taxon.unsqueeze(0), species_probs
                )
                clip_log_probs = torch.log(clip_probs + 1e-12)

                date_log_probs = F.log_softmax(network(preprocessed_dates[i].unsqueeze(0).to(device)), dim=-1)

                alpha = 0.7
                fused_log_probs = F.log_softmax(alpha * clip_log_probs + (1 - alpha) * date_log_probs - log_prior, dim=-1)

            clip_pred  = clip_log_probs.argmax().item()
            fused_pred = fused_log_probs.argmax().item()
            print(f"CLIP: {unique_labels[clip_pred]!r}, Fused: {unique_labels[fused_pred]!r}, Actual: {unique_labels[true_label]!r}")
            clip_correct  += int(clip_pred  == true_label)
            fused_correct += int(fused_pred == true_label)
            fused_confidences.append(fused_log_probs[0, fused_pred].exp().item())

        print(f"\nCLIP-only accuracy:   {clip_correct  / len(test_labels):.3f}")
        print(f"Fused accuracy:       {fused_correct / len(test_labels):.3f}")
        print(f"Avg fused confidence: {np.mean(fused_confidences):.4f}")


    # Run the chosen experiment(s)
    if experiment in ("location", "both"):
        network   = LocToTaxon(output_dim=len(unique_labels)).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        loss_fn   = nn.CrossEntropyLoss()

        for epoch in range(10):
            total_loss, steps = 0.0, 0
            for batch in train_loader:
                x = batch["coords"].to(device, non_blocking=True)
                y = batch["label_idx"].to(device, non_blocking=True)
                optimizer.zero_grad()
                l = loss_fn(network(x), y)
                l.backward()
                optimizer.step()
                total_loss += l.item(); steps += 1
            print(f"Epoch {epoch} | loss: {total_loss/steps:.4f}")

        network.eval()
        clip_correct, fused_correct, fused_confidences = 0, 0, []

        for i, img in enumerate(preprocessed_images):
            img = img.to(device)
            true_label = test_labels[i]

            with torch.no_grad(), torch.amp.autocast(device.type):
                image_features = model.encode_image(img)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # score at species level, then sum probs per taxon group
                species_probs = F.softmax(100 * image_features @ species_text_embs.T, dim=-1).float()
                clip_probs    = torch.zeros(1, len(unique_labels), device=device).scatter_add_(
                    1, species_to_taxon.unsqueeze(0), species_probs
                )
                clip_log_probs = torch.log(clip_probs + 1e-12)

                coord_log_probs = F.log_softmax(network(coords[i].unsqueeze(0).to(device)), dim=-1)

                # alpha=0.5 gives equal weight 
                alpha = 0.5
                fused_log_probs = F.log_softmax(alpha * clip_log_probs + (1 - alpha) * coord_log_probs - log_prior, dim=-1)

            clip_pred  = clip_log_probs.argmax().item()
            fused_pred = fused_log_probs.argmax().item()
            print(f"CLIP: {unique_labels[clip_pred]!r}, Fused: {unique_labels[fused_pred]!r}, Actual: {unique_labels[true_label]!r}")
            clip_correct  += int(clip_pred  == true_label)
            fused_correct += int(fused_pred == true_label)
            fused_confidences.append(fused_log_probs[0, fused_pred].exp().item())

        print(f"\nCLIP-only accuracy:   {clip_correct  / len(test_labels):.3f}")
        print(f"Fused accuracy:       {fused_correct / len(test_labels):.3f}")
        print(f"Avg fused confidence: {np.mean(fused_confidences):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="taxonomy_experiment")
    parser.add_argument("--e",     choices=["date", "location", "both"], required=True)
    parser.add_argument("--level", choices=TAXONOMY_LEVELS, default="scientificName")
    args = parser.parse_args()

    run_experiment(args.e, args.level)