import os
import joblib
import numpy as np
import scipy.linalg
import mne
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

mne.set_log_level("ERROR")

MOTOR_CHANNELS = [
    'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
    'C5',  'C3',  'C1',  'CZ',  'C2',  'C4',  'C6',
    'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
    'P1',  'PZ',  'P2',  'OZ'
]

def inv_sqrt_m(cov_mat: np.ndarray) -> np.ndarray:
    evals, evecs = scipy.linalg.eigh(cov_mat)
    evals = np.maximum(evals, 1e-10)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

def extract_subject_epochs(subject_id: int, runs: list):
    X_sub, y_sub = [], []
    for run in runs:
        try:
            edf_path = mne.datasets.eegbci.load_data(subject_id, run, update_path=True, verbose=False)[0]
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            raw.rename_channels(lambda x: x.strip('.').strip().upper())
            
            avail = [ch for ch in MOTOR_CHANNELS if ch in raw.ch_names]
            if len(avail) != 21:
                continue
            raw.pick_channels(avail, ordered=True)
            raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', verbose=False)
            
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            t1, t2 = event_id.get('T1'), event_id.get('T2')
            
            valid_events, labels = [], []
            for e in events:
                if e[2] == t1:
                    valid_events.append(e)
                    labels.append(0)
                elif e[2] == t2:
                    valid_events.append(e)
                    labels.append(1)
                    
            if valid_events:
                # Crop to active motor imagery window: 0.5s to 2.5s post-cue
                epochs = mne.Epochs(
                    raw, np.array(valid_events), tmin=0.5, tmax=2.5,
                    baseline=None, preload=True, verbose=False
                )
                data = epochs.get_data()
                X_sub.append(data)
                y_sub.extend(labels)
        except Exception:
            continue
            
    if X_sub:
        return np.concatenate(X_sub, axis=0), np.array(y_sub)
    return None, None

def train_and_save_global_model():
    os.makedirs("models", exist_ok=True)
    model_path = "models/global_riemann_model.joblib"
    
    print("[1/4] Extracting reaction-latency cropped epochs (t=0.5s to 2.5s)...", flush=True)
    X_list, y_list, groups_list = [], [], []
    
    for sub in range(1, 61):
        X_s, y_s = extract_subject_epochs(sub, runs=[4, 8, 12])  # Imagery runs
        if X_s is not None:
            X_list.append(X_s)
            y_list.append(y_s)
            groups_list.extend([sub] * len(y_s))
            
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.array(groups_list)
    
    n_trials, n_channels, n_samples = X.shape
    print(f"      Loaded {n_trials} trials across {len(np.unique(groups))} subjects ({n_samples} samples/trial).", flush=True)
    
    print("[2/4] Estimating Covariances & Applying Per-Subject ESA...", flush=True)
    cov_estimator = Covariances(estimator='lwf')
    C_raw = cov_estimator.fit_transform(X)
    
    C_aligned = np.zeros_like(C_raw)
    for sub in np.unique(groups):
        sub_mask = (groups == sub)
        C_sub = C_raw[sub_mask]
        R_sub = np.mean(C_sub, axis=0)
        inv_R = inv_sqrt_m(R_sub)
        
        for idx in np.where(sub_mask)[0]:
            C_aligned[idx] = inv_R @ C_raw[idx] @ inv_R

    print("[3/4] Fitting Riemannian Tangent Space & Classifier...", flush=True)
    ts = TangentSpace(metric='riemann')
    T_all = ts.fit_transform(C_aligned)
    
    scaler = StandardScaler()
    T_scaled = scaler.fit_transform(T_all)
    
    clf = LogisticRegression(C=0.1, solver='lbfgs', class_weight='balanced', max_iter=1000)
    clf.fit(T_scaled, y)
    
    model_artifact = {
        'cov_estimator': cov_estimator,
        'tangent_space': ts,
        'scaler': scaler,
        'classifier': clf,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'classes': ['Left Hand', 'Right Hand']
    }
    joblib.dump(model_artifact, model_path)
    print(f"--> Global model successfully saved to '{model_path}'.")

if __name__ == "__main__":
    train_and_save_global_model()