import numpy as np
import torch
 
 
def encode_date(month, day):
    """
    Encode month and day as cyclical features using sin/cos projections.
 
    Each component maps to a point on a unit circle so that boundary values
    wrap smoothly (e.g. December → January, day 365 → day 1).
    Output is scaled from [-1, 1] to [0, 1] for compatibility with ReLU nets.
    """
    month_norm       = (month - 1) / 12
    day_of_year_norm = ((month - 1) * 30 + day) / 365
 
    month_sin = (np.sin(2 * np.pi * month_norm)       + 1) / 2
    month_cos = (np.cos(2 * np.pi * month_norm)       + 1) / 2
    day_sin   = (np.sin(2 * np.pi * day_of_year_norm) + 1) / 2
    day_cos   = (np.cos(2 * np.pi * day_of_year_norm) + 1) / 2
 
    return torch.tensor([month_sin, month_cos, day_sin, day_cos], dtype=torch.float32)
 