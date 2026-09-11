import argparse
import joblib
import numpy as np
import scipy.linalg
import mne

from predict import extract_trials_and_labels, inv_sqrt_m, resolve_edf_path

mne.set_log_level("ERROR")

def evaluate_full_subject(subject_id: int, model_path: str = "models/global_riemann_model.joblib"):
    artifact = joblib.load(model_path)
    cov_est = artifact['cov_estimator']
    ts = artifact['tangent_space']
    scaler = artifact.get('scaler')
    clf = artifact['classifier']
    target_samples = artifact.get('n_samples', 320)
    
    imagery_runs = [4, 8, 12]
    all_preds, all_labels = [], []
    
    # Collect calibration covariance (Run 3: Motor Execution)
    calib_edf = resolve_edf_path(subject=subject_id, run=3)
    calib_trials, _ = extract_trials_and_labels(calib_edf, target_samples)
    C_calib = cov_est.transform(calib_trials)
    R_sub = np.mean(C_calib, axis=0)
    inv_R = inv_sqrt_m(R_sub)
    
    for run in imagery_runs:
        edf_path = resolve_edf_path(subject=subject_id, run=run)
        trials_data, true_labels = extract_trials_and_labels(edf_path, target_samples)
        
        C_trials = cov_est.transform(trials_data)
        C_aligned = np.array([inv_R @ c @ inv_R for c in C_trials])
        
        tangents = ts.transform(C_aligned)
        if scaler is not None:
            tangents = scaler.transform(tangents)
            
        preds = clf.predict(tangents)
        all_preds.extend(preds)
        all_labels.extend(true_labels)
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = np.mean(all_preds == all_labels) * 100
    
    print(f"\n==================================================")
    print(f"   FULL SUBJECT {subject_id:03d} BENCHMARK (45 IMAGERY TRIALS)   ")
    print(f"==================================================")
    print(f" Total Trials Evaluated : {len(all_labels)}")
    print(f" Total Correct         : {np.sum(all_preds == all_labels)}/{len(all_labels)}")
    print(f" Overall Accuracy      : {acc:.2f}%")
    print(f"==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=14)
    args = parser.parse_args()
    evaluate_full_subject(args.subject)