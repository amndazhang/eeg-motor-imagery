import os
import argparse
import joblib
import numpy as np
import scipy.linalg
import mne

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

def resolve_edf_path(input_path: str = None, subject: int = None, run: int = None) -> str:
    if input_path and os.path.exists(input_path):
        return input_path
    if subject is not None and run is not None:
        paths = mne.datasets.eegbci.load_data(subject, run, update_path=True, verbose=False)
        return paths[0]
    if input_path:
        base = os.path.basename(input_path)
        if base.startswith('S') and 'R' in base:
            try:
                sub_str, rest = base[1:].split('R')
                run_str = rest.split('.')[0]
                paths = mne.datasets.eegbci.load_data(int(sub_str), int(run_str), update_path=True, verbose=False)
                return paths[0]
            except Exception:
                pass
    raise FileNotFoundError(f"Could not locate EDF file for: {input_path or (subject, run)}")

def extract_trials_and_labels(edf_path: str, target_samples: int = 320):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw.rename_channels(lambda x: x.strip('.').strip().upper())
    
    avail_channels = [ch for ch in MOTOR_CHANNELS if ch in raw.ch_names]
    if len(avail_channels) != len(MOTOR_CHANNELS):
        missing = set(MOTOR_CHANNELS) - set(avail_channels)
        raise ValueError(f"EDF missing channels: {missing}")
        
    raw.pick_channels(avail_channels, ordered=True)
    raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', verbose=False)
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    t1_code, t2_code = event_id.get('T1'), event_id.get('T2')
    
    valid_events, labels = [], []
    for e in events:
        if e[2] == t1_code:
            valid_events.append(e)
            labels.append(0)
        elif e[2] == t2_code:
            valid_events.append(e)
            labels.append(1)
            
    if len(valid_events) > 0:
        # Crop window to t = 0.5s -> 2.5s post-cue
        epochs = mne.Epochs(
            raw, np.array(valid_events), tmin=0.5, tmax=2.5,
            baseline=None, preload=True, verbose=False
        )
        data = epochs.get_data()
    else:
        raw_data = raw.get_data()
        n_trials = raw_data.shape[1] // target_samples
        data = np.array([raw_data[:, i*target_samples:(i+1)*target_samples] for i in range(n_trials)])
        labels = [-1] * n_trials

    if data.shape[2] != target_samples:
        if data.shape[2] > target_samples:
            data = data[:, :, :target_samples]
        else:
            pad_s = target_samples - data.shape[2]
            data = np.pad(data, ((0, 0), (0, 0), (0, pad_s)), mode='edge')
            
    return data, np.array(labels)

def predict_trials(model_path: str, input_path: str = None, subject: int = None, run: int = None, calib_path: str = None):
    resolved_edf = resolve_edf_path(input_path, subject, run)
    
    artifact = joblib.load(model_path)
    cov_est = artifact['cov_estimator']
    ts = artifact['tangent_space']
    scaler = artifact.get('scaler')
    clf = artifact['classifier']
    target_samples = artifact.get('n_samples', 320)
    
    trials_data, true_labels = extract_trials_and_labels(resolved_edf, target_samples)
    
    C_trials = cov_est.transform(trials_data)
    
    if calib_path:
        resolved_calib = resolve_edf_path(calib_path)
        calib_trials, _ = extract_trials_and_labels(resolved_calib, target_samples)
        C_calib = cov_est.transform(calib_trials)
        R_sub = np.mean(C_calib, axis=0)
        mode_str = f"Few-Shot Calibrated ({os.path.basename(resolved_calib)})"
    else:
        R_sub = np.mean(C_trials, axis=0)
        mode_str = "Zero-Shot (Uncalibrated)"
        
    inv_R = inv_sqrt_m(R_sub)
    C_aligned = np.array([inv_R @ c @ inv_R for c in C_trials])
    
    tangents = ts.transform(C_aligned)
    if scaler is not None:
        tangents = scaler.transform(tangents)
        
    probs = clf.predict_proba(tangents)
    preds = np.argmax(probs, axis=1)
    
    class_names = artifact['classes']
    
    print("\n-------------------------------------------------------------------------")
    print("                 REAL-TIME BCI INFERENCE EVALUATION                      ")
    print("-------------------------------------------------------------------------")
    print(f" Source File     : {os.path.basename(resolved_edf)}")
    print(f" Alignment Mode  : {mode_str}")
    print(f" Total Trials    : {len(preds)}")
    print("-------------------------------------------------------------------------")
    
    correct_count = 0
    has_ground_truth = (true_labels[0] != -1)
    
    for idx, (p, prob) in enumerate(zip(preds, probs)):
        pred_str = class_names[p]
        conf = prob[p] * 100
        
        if has_ground_truth:
            actual_str = class_names[true_labels[idx]]
            is_correct = (p == true_labels[idx])
            if is_correct:
                correct_count += 1
            status = "MATCH" if is_correct else "MISMATCH"
            print(f" Trial {idx+1:02d} | Actual: {actual_str:<10} | Pred: {pred_str:<10} | Conf: {conf:.1f}% | {status}")
        else:
            print(f" Trial {idx+1:02d} | Pred: {pred_str:<10} | Conf: {conf:.1f}%")
            
    print("-------------------------------------------------------------------------")
    if has_ground_truth:
        acc = (correct_count / len(preds)) * 100
        print(f" Trial Accuracy  : {acc:.1f}% ({correct_count}/{len(preds)} Correct)")
        print("-------------------------------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time BCI Motor Imagery Prediction CLI")
    parser.add_argument("--input", required=False, help="Path or filename of target trial EDF")
    parser.add_argument("--subject", type=int, required=False, help="Subject ID (1-109)")
    parser.add_argument("--run", type=int, required=False, help="Run ID (4, 8, or 12 for imagery)")
    parser.add_argument("--calib", required=False, help="Optional calibration EDF filename/path")
    parser.add_argument("--model", default="models/global_riemann_model.joblib", help="Path to trained model artifact")
    
    args = parser.parse_args()
    predict_trials(args.model, args.input, args.subject, args.run, args.calib)