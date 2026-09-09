import numpy as np
import scipy.linalg
from src.preprocessing import preprocess_subject

def apply_euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """
    Applies Euclidean Space Alignment (ESA) across trials to regularize sample covariance.
    """
    covs = np.array([np.cov(x) for x in X])
    mean_cov = np.mean(covs, axis=0)
    evals, evecs = scipy.linalg.eigh(mean_cov)
    evals = np.maximum(evals, 1e-10)
    inv_sqrt_cov = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    
    X_aligned = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_aligned[i] = inv_sqrt_cov @ X[i]
    return X_aligned

def load_dataset(
    subject_ids: list, 
    runs: list = [3, 7, 11], 
    motor_only: bool = True, 
    use_esa: bool = True
):
    """
    Loads and concatenates cleaned trials across specified subject IDs and runs.
    """
    X_list, y_list, sub_list = [], [], []
    
    for sub in subject_ids:
        try:
            epochs = preprocess_subject(sub, runs=runs, motor_only=motor_only)
            X_sub = epochs.get_data(copy=True)
            y_sub = epochs.events[:, -1] - 1  # Convert 1, 2 to 0, 1
            
            if use_esa:
                X_sub = apply_euclidean_alignment(X_sub)
                
            X_list.append(X_sub)
            y_list.append(y_sub)
            sub_list.append(np.full(len(y_sub), sub))
        except Exception as e:
            print(f"Warning: Failed to load subject {sub} — {e}")
            
    if not X_list:
        raise ValueError("No subject data could be loaded.")
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subjects = np.concatenate(sub_list, axis=0)
    
    return X, y, subjects