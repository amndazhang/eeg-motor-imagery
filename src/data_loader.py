import numpy as np
import scipy.linalg
from src.preprocessing import preprocess_subject

def apply_euclidean_alignment(X: np.ndarray) -> np.ndarray:
    covs = np.array([np.cov(x) for x in X])
    mean_cov = np.mean(covs, axis=0)
    evals, evecs = scipy.linalg.eigh(mean_cov)
    evals = np.maximum(evals, 1e-10)
    inv_sqrt_cov = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    
    X_aligned = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_aligned[i] = inv_sqrt_cov @ X[i]
    return X_aligned

def extract_sliding_windows(
    X: np.ndarray, 
    y: np.ndarray, 
    groups: np.ndarray, 
    sfreq: float = 160.0, 
    win_sec: float = 2.0, 
    step_sec: float = 0.5
):
    """
    Slices trial array X (n_trials, n_channels, n_times) into overlapping temporal windows.
    Triples training samples while maintaining exact target alignment.
    """
    win_len = int(win_sec * sfreq)
    step_len = int(step_sec * sfreq)
    n_samples = X.shape[-1]
    
    X_wins, y_wins, groups_wins = [], [], []
    
    for start in range(0, n_samples - win_len + 1, step_len):
        end = start + win_len
        X_slice = X[:, :, start:end]
        
        X_wins.append(X_slice)
        y_wins.append(y)
        groups_wins.append(groups)
        
    X_augmented = np.concatenate(X_wins, axis=0)
    y_augmented = np.concatenate(y_wins, axis=0)
    groups_augmented = np.concatenate(groups_wins, axis=0)
    
    return X_augmented, y_augmented, groups_augmented

def load_dataset(
    subject_ids: list, 
    runs: list = [4, 8, 12], 
    motor_only: bool = True, 
    use_esa: bool = True,
    use_sliding_window: bool = True
):
    X_list, y_list, sub_list = [], [], []
    
    for sub in subject_ids:
        try:
            epochs = preprocess_subject(sub, runs=runs, motor_only=motor_only)
            X_sub = epochs.get_data(copy=True)
            y_sub = epochs.events[:, -1] - 1
            
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
    
    if use_sliding_window:
        X, y, subjects = extract_sliding_windows(X, y, subjects)
        
    return X, y, subjects