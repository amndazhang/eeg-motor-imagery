import numpy as np
import scipy.linalg
from typing import Tuple, List
from src.preprocessing import preprocess_subject

def apply_euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """
    Normalizes a subject's mean trial covariance matrix to the identity matrix.
    X shape: (n_epochs, n_channels, n_times)
    """
    covs = np.array([np.cov(x) for x in X])
    mean_cov = np.mean(covs, axis=0)
    
    # Compute R^(-1/2) via eigendecomposition
    evals, evecs = scipy.linalg.eigh(mean_cov)
    evals = np.maximum(evals, 1e-10)  # Numerical stability
    inv_sqrt_cov = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    
    # Project trials: X_aligned = R^(-1/2) * X
    X_aligned = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_aligned[i] = inv_sqrt_cov @ X[i]
        
    return X_aligned

def load_dataset(
    subject_ids: List[int], 
    runs: List[int] = [4, 8, 12],
    motor_only: bool = True,
    use_esa: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads EEG data across subjects with optional Euclidean Space Alignment.
    """
    X_list, y_list, groups_list = [], [], []

    for sub_id in subject_ids:
        try:
            epochs = preprocess_subject(subject_id=sub_id, runs=runs, motor_only=motor_only)
            X_sub = epochs.get_data(copy=True)
            
            # Apply Euclidean Alignment per subject before cross-subject pooling
            if use_esa:
                X_sub = apply_euclidean_alignment(X_sub)
            
            raw_labels = epochs.events[:, -1]
            t1_val = epochs.event_id['T1']
            y_sub = np.where(raw_labels == t1_val, 0, 1)
            
            X_list.append(X_sub)
            y_list.append(y_sub)
            groups_list.extend([sub_id] * len(y_sub))
            
        except Exception as e:
            print(f"Warning: Failed to load subject {sub_id} — {e}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.array(groups_list)

    return X, y, groups