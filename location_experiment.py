
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from experimental_setup import setup

class Loc_to_species_Common(nn.Module):
    """Maps 2-dim location features into one of Cryptic Bio species Scientific Name"""
    def __init__(self, output_dim=158): # 158 unique species in the common subset
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),    
        )

    def forward(self, x):
       x=torch.flatten(x, start_dim=1)  # Flatten the input to (batch_size, coord_dim)
       return self.network(x)
        
# The main training loop for the location-based experiment
def run():
    ctx = setup()
 
    device              = ctx["device"]
    model               = ctx["model"]
    train_loader        = ctx["train_loader"]
    unique_names        = ctx["unique_names"]
    species_text_embs   = ctx["species_text_embs"]
    log_prior           = ctx["log_prior"]
    preprocessed_images = ctx["preprocessed_images"]
    coords              = ctx["coords"]
    test_labels         = ctx["test_labels"]
    
    
    network_coords_to_species = Loc_to_species_Common(output_dim=len(unique_names)).to(device)
    loss=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network_coords_to_species.parameters(), lr=1e-3)
    for epoch in range(10):
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            x = batch['coords'].to(device, non_blocking=True)
            y = batch['label_idx'].to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = network_coords_to_species(x)
            l = loss(logits, y)
            l.backward()
            optimizer.step()
            total_loss += l.item(); steps += 1
        print(f"Epoch {epoch} | train loss: {total_loss/steps:.4f}")
        

    network_coords_to_species.eval()
    species_text_embs_gpu = species_text_embs.to(device)   

    clip_correct_2 = 0
    fused_correct_2 = 0
    fused_confidences_2 = []
    total = 0

    # Fuse CLIP and location predictions via weighted log-prob average.
    for i, img in enumerate(preprocessed_images):
        img = img.to(device)
        true_label = test_labels[i] 

        with torch.no_grad(), torch.amp.autocast(device.type):
            #  CLIP branch 
            image_features = model.encode_image(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            clip_logits = 100 * image_features @ species_text_embs_gpu.T
            clip_log_probs = F.log_softmax(clip_logits.float(), dim=-1)        # normalize between models

            #  Coordinates branch 
            coord_input = coords[i].unsqueeze(0).to(device)
            coord_logits = network_coords_to_species(coord_input)                   
            coord_log_probs = F.log_softmax(coord_logits.float(), dim=-1)                

            alpha = 0.5  # location signal is stronger than date so 0.5 beats date_experiment's 0.7
            fused_log_probs = alpha * clip_log_probs + (1 - alpha) * coord_log_probs - log_prior
            fused_log_probs = F.log_softmax(fused_log_probs.float(), dim=-1)           

            clip_pred  = clip_log_probs.argmax(dim=-1).item()
            fused_pred = fused_log_probs.argmax(dim=-1).item()
            fused_prob = fused_log_probs[0, fused_pred].exp().item()

        clip_correct_2  += int(clip_pred  == true_label)
        fused_correct_2 += int(fused_pred == true_label)
        fused_confidences_2.append(fused_prob)
        total += 1

        # print(f"Image {i}: CLIP={unique_names[clip_pred]}, "
        #     f"Fused={unique_names[fused_pred]} (p={fused_prob:.3f}), "
        #     f"True={unique_names[true_label]}")

    print(f"\nCLIP-only accuracy:  {clip_correct_2/total:.3f}")
    print(f"Fused accuracy:      {fused_correct_2/total:.3f}")
    print(f"Avg fused confidence: {np.mean(fused_confidences_2):.4f}")