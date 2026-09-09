import argparse
import json
import ssl
from pathlib import Path
import joblib
import numpy as np
import scipy.linalg
import mne
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import read_raw_edf

ssl._create_default_https_context = ssl._create_unverified_context

MOTOR_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
]

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

def predict_edf(edf_path: str, model_path: str = "models/global_csp_lda.joblib") -> dict:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train script first.")
        
    model = joblib.load(model_path)
    raw = read_raw_edf(edf_path, preload=True, verbose=False)
    
    if hasattr(eegbci, 'standardize'):
        eegbci.standardize(raw)
    else:
        raw.rename_channels(lambda x: x.strip('.').strip())
        
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, verbose=False)
    
    valid_channels = [ch for ch in MOTOR_CHANNELS if ch in raw.ch_names]
    raw.pick(valid_channels)
    
    raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', verbose=False)
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_event_id = {k: v for k, v in event_id.items() if k in ['T1', 'T2']}
    
    if not target_event_id:
        raise ValueError("No T1 or T2 events found in the provided EDF file.")
        
    epochs = mne.Epochs(
        raw, events=events, event_id=target_event_id, 
        tmin=0.5, tmax=3.5, baseline=None, preload=True, verbose=False
    )
    
    X = epochs.get_data(copy=True)
    
    # Align incoming test session covariance
    X_aligned = apply_euclidean_alignment(X)
    
    preds = model.predict(X_aligned).tolist()
    probs = model.predict_proba(X_aligned).tolist()
    
    return {
        "num_trials": len(preds),
        "predictions": preds,
        "class_mapping": {"0": "Left Fist", "1": "Right Fist"},
        "probabilities": probs
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Motor Imagery from raw EDF file.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw EDF file")
    parser.add_argument("--model", type=str, default="models/global_csp_lda.joblib", help="Path to saved joblib model")
    
    args = parser.parse_args()
    results = predict_edf(args.input, args.model)
    print(json.dumps(results, indent=2))