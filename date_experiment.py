import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from experimental_setup import setup
BATCH = 256
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Experiment 1 - adding Date (month + day) to the existent embeddings 
#Encoding the date(month,day), using sine and cosine following the intuition from: 
#https://harrisonpim.com/blog/the-best-way-to-encode-dates-times-and-other-cyclical-features . 
# We kinda mimic the first stage of the BioClip training - learing meaningful represemntation of date


class Date_to_species_Common(nn.Module):
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
        x = torch.flatten(x, start_dim=1)  # Flatten the input to (batch_size, date_dim)
        return self.network(x)
        
# The main training loop
def run():
    ctx = setup()
 
    device              = ctx["device"]
    model               = ctx["model"]
    train_loader        = ctx["train_loader"]
    unique_names        = ctx["unique_names"]
    species_text_embs   = ctx["species_text_embs"]
    log_prior           = ctx["log_prior"]
    preprocessed_images = ctx["preprocessed_images"]
    preprocessed_dates  = ctx["preprocessed_dates"]
    test_labels         = ctx["test_labels"]
    processed           = ctx["processed"]
    name_to_idx         = ctx["name_to_idx"]

    network_date_to_species = Date_to_species_Common(
        output_dim=len(unique_names)
    ).to(device)
    
    loss   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network_date_to_species.parameters(), lr=1e-3)
    
    for epoch in range(10):
        network_date_to_species.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            x = batch['encoded_input'].to(device, non_blocking=True)
            y = batch['label_idx'].to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = network_date_to_species(x)
            l = loss(logits, y)
            l.backward()
            optimizer.step()
            total_loss += l.item(); steps += 1
        print(f"Epoch {epoch} | train loss: {total_loss/steps:.4f}")

    # scheduler.step()
    labels = processed[:]['label_idx']  
    counts = torch.bincount(labels, minlength=len(name_to_idx)).float()
    p = counts / counts.sum()
    marginal_loss = -(p * torch.log(p + 1e-12)).sum().item()
    print(marginal_loss)

   
    species_text_embs_gpu = species_text_embs.to(device)
    prob_list, clip_baseline_correct = [], 0
 
    # for i, img in enumerate(preprocessed_images):
    #     img = img.to(device)
    #     with torch.no_grad(), torch.amp.autocast(device.type):
    #         image_features = model.encode_image(img)
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #         logit_scale = model.logit_scale.exp()
    #         text_probs    = (logit_scale * image_features @ species_text_embs_gpu.T).softmax(dim=-1)
    #         predicted_idx = text_probs.argmax().item()
    #         if predicted_idx == test_labels[i]:
    #             clip_baseline_correct += 1
    #         prob_list.append(text_probs[0, predicted_idx].item())
    #         print(
    #             f"Test Image {i:04d}: pred={unique_names[predicted_idx]!r}  "
    #             f"p={text_probs[0, predicted_idx].item():.4f}  "
    #             f"correct={predicted_idx == test_labels[i]}"
    #         )
 
    # print(f"Avg predicted probability: {np.mean(prob_list):.4f}")
    # print(f"CLIP-only correct: {clip_baseline_correct}/{len(test_labels)}")


    # network_date_to_species.eval()
    # species_text_embs_gpu = species_text_embs.to(device)   


    clip_correct = 0
    fused_correct = 0
    fused_confidences = []
    total = 0

    # Fuse CLIP + date predictions in log-space (subtract log_prior to debias class frequencies).
    # Track both CLIP-only and fused accuracy.
    for i, img in enumerate(preprocessed_images):
        img = img.to(device)
        true_label = test_labels[i]   

        with torch.no_grad(), torch.amp.autocast(device.type):
            #  CLIP branch 
            image_features = model.encode_image(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            clip_logits = 100 * image_features @ species_text_embs_gpu.T
            clip_log_probs = F.log_softmax(clip_logits.float(), dim=-1)        # normalize between models

            #  Date branch 
            date_input = preprocessed_dates[i].unsqueeze(0).to(device)
            date_logits = network_date_to_species(date_input) 
            date_log_probs = F.log_softmax(date_logits.float(), dim=-1)                

            alpha = 0.7
            fused_log_probs = (alpha * clip_log_probs) + ((1 - alpha) * date_log_probs) - log_prior
            fused_log_probs = F.log_softmax(fused_log_probs.float(), dim=-1)           

            clip_pred  = clip_log_probs.argmax(dim=-1).item()
            fused_pred = fused_log_probs.argmax(dim=-1).item()
            fused_prob = fused_log_probs[0, fused_pred].exp().item()

        clip_correct  += int(clip_pred  == true_label)
        fused_correct += int(fused_pred == true_label)
        fused_confidences.append(fused_prob)
        total += 1

        # print(f"Image {i}: CLIP={unique_names[clip_pred]}, "
        #     f"Fused={unique_names[fused_pred]} (p={fused_prob:.3f}), "
        #     f"True={unique_names[true_label]}")

    print(f"\nCLIP-only accuracy:  {clip_correct/total:.3f}")
    print(f"Fused accuracy:      {fused_correct/total:.3f}")
    print(f"Avg fused confidence: {np.mean(fused_confidences):.4f}")
