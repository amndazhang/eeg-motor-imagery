import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from src.data_loader import load_dataset
from src.models import FilterBankTangentSpace

def run_fb_riemann_few_shot():
    test_subjects = list(range(1, 61))
    print("Loading raw trial data for 60 subjects (full 3.0s trials)...", flush=True)
    
    X_raw, y_raw, groups = load_dataset(
        subject_ids=test_subjects, 
        runs=[4, 8, 12], 
        motor_only=True, 
        use_esa=True,
        use_sliding_window=False
    )
    
    print("\nExtracting Filter Bank Tangent features across 6 frequency sub-bands...", flush=True)
    fb_transformer = FilterBankTangentSpace(sfreq=160.0)
    T_all = fb_transformer.fit_transform(X_raw)  # Matrix shape: (N_trials, 1386)
    
    print(f"Extracted {T_all.shape[1]} spatio-spectral tangent features per trial.", flush=True)
    
    logo = LeaveOneGroupOut()
    
    for k_shots in [2, 4, 6]:
        scores = []
        print(f"\nEvaluating FBRiemann Few-Shot Calibration for k={k_shots} trials...", flush=True)
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(T_all, y_raw, groups=groups)):
            T_train, y_train = T_all[train_idx], y_raw[train_idx]
            T_test_sub, y_test_sub = T_all[test_idx], y_raw[test_idx]
            
            idx_c0 = np.where(y_test_sub == 0)[0]
            idx_c1 = np.where(y_test_sub == 1)[0]
            half_k = k_shots // 2
            
            if len(idx_c0) < half_k or len(idx_c1) < half_k:
                continue
                
            calib_idx = np.concatenate([idx_c0[:half_k], idx_c1[:half_k]])
            eval_idx = np.setdiff1d(np.arange(len(y_test_sub)), calib_idx)
            
            T_calib, y_calib = T_test_sub[calib_idx], y_test_sub[calib_idx]
            T_eval, y_eval = T_test_sub[eval_idx], y_test_sub[eval_idx]
            
            # Select top 120 most informative spatio-spectral features
            selector = SelectKBest(score_func=mutual_info_classif, k=120)
            T_train_sel = selector.fit_transform(T_train, y_train)
            T_calib_sel = selector.transform(T_calib)
            T_eval_sel = selector.transform(T_eval)
            
            # Train global classifier
            clf = LogisticRegression(C=0.15, solver='lbfgs', max_iter=500)
            clf.fit(T_train_sel, y_train)
            
            # Fine-tune intercept shift on target user calibration samples
            logits_calib = (T_calib_sel @ clf.coef_.T + clf.intercept_).ravel()
            calib_labels_signed = np.where(y_calib == 1, 1.0, -1.0)
            bias_shift = np.mean(calib_labels_signed - logits_calib) * 0.15
            
            logits_eval = (T_eval_sel @ clf.coef_.T + (clf.intercept_ + bias_shift)).ravel()
            preds = np.where(logits_eval >= 0, 1, 0)
            
            scores.append(np.mean(preds == y_eval))
            
            if (fold + 1) % 20 == 0:
                print(f"  Processed {fold + 1}/60 subjects...", flush=True)
                
        print(f"--> FBRiemann Few-Shot Accuracy (k={k_shots} Calibration Trials): {np.mean(scores) * 100:.1f}%")

if __name__ == "__main__":
    run_fb_riemann_few_shot()