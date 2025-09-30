import numpy as np
from scipy.stats import ks_2samp

def check_model_drift(new_preds, old_labels):
    ks_stat, p_value = ks_2samp(new_preds, old_labels)
    if p_value < 0.05:
        print("Model drift detected!")
    else:
        print("No significant drift.")
