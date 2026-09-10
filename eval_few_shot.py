import numpy as np
import scipy.linalg
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from src.data_loader import load_dataset

def inv_sqrt_m(cov_mat: np.ndarray) -> np.ndarray:
    """Computes R^(-1/2) for Euclidean Space Alignment."""
    evals, evecs = scipy.linalg.eigh(cov_mat)
    evals = np.maximum(evals, 1e-10)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

def run_few_shot_benchmark():
    test_subjects = list(range(1, 61))
    print("Loading raw trial data for 60 subjects...", flush=True)
    
    X_raw, y_raw, groups = load_dataset(
        subject_ids=test_subjects, 
        runs=[4, 8, 12], 
        motor_only=True, 
        use_esa=False,
        use_sliding_window=False
    )
    
    print("Pre-calculating covariance matrices for all trials...", flush=True)
    cov_estimator = Covariances(estimator='lwf')
    C_all = cov_estimator.fit_transform(X_raw)  # Matrix shape: (N_trials, 21, 21)
    
    logo = LeaveOneGroupOut()
    
    for k_shots in [2, 4, 6]:
        scores = []
        print(f"\nEvaluating Few-Shot Calibration for k={k_shots} trials...", flush=True)
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(C_all, y_raw, groups=groups)):
            C_train, y_train = C_all[train_idx], y_raw[train_idx]
            C_test_sub, y_test_sub = C_all[test_idx], y_raw[test_idx]
            
            idx_c0 = np.where(y_test_sub == 0)[0]
            idx_c1 = np.where(y_test_sub == 1)[0]
            half_k = k_shots // 2
            
            if len(idx_c0) < half_k or len(idx_c1) < half_k:
                continue
                
            calib_idx = np.concatenate([idx_c0[:half_k], idx_c1[:half_k]])
            eval_idx = np.setdiff1d(np.arange(len(y_test_sub)), calib_idx)
            
            C_calib, y_calib = C_test_sub[calib_idx], y_test_sub[calib_idx]
            C_eval, y_eval = C_test_sub[eval_idx], y_test_sub[eval_idx]
            
            # 1. Global training alignment reference
            R_global = np.mean(C_train, axis=0)
            inv_R_global = inv_sqrt_m(R_global)
            
            # Direct covariance matrix alignment
            C_train_aligned = np.array([inv_R_global @ c @ inv_R_global for c in C_train])
            
            # 2. Fit global Tangent Space Classifier
            ts = TangentSpace(metric='riemann')
            T_train = ts.fit_transform(C_train_aligned)
            
            clf = LogisticRegression(C=0.1, solver='lbfgs', max_iter=500)
            clf.fit(T_train, y_train)
            
            # 3. Target calibration covariance (blended 50/50 with global mean for matrix stability)
            R_calib = np.mean(C_calib, axis=0)
            R_target = 0.5 * R_global + 0.5 * R_calib
            inv_R_target = inv_sqrt_m(R_target)
            
            C_calib_aligned = np.array([inv_R_target @ c @ inv_R_target for c in C_calib])
            C_eval_aligned = np.array([inv_R_target @ c @ inv_R_target for c in C_eval])
            
            T_calib = ts.transform(C_calib_aligned)
            T_eval = ts.transform(C_eval_aligned)
            
            # 4. Fine-tune intercept shift on calibration logits without touching directional weights
            logits_calib = (T_calib @ clf.coef_.T + clf.intercept_).ravel()
            calib_labels_signed = np.where(y_calib == 1, 1.0, -1.0)
            bias_shift = np.mean(calib_labels_signed - logits_calib) * 0.15
            
            logits_eval = (T_eval @ clf.coef_.T + (clf.intercept_ + bias_shift)).ravel()
            preds = np.where(logits_eval >= 0, 1, 0)
            
            scores.append(np.mean(preds == y_eval))
            
            if (fold + 1) % 20 == 0:
                print(f"  Processed {fold + 1}/60 subjects...", flush=True)
                
        print(f"--> Few-Shot Accuracy (k={k_shots} Calibration Trials): {np.mean(scores) * 100:.1f}%")

if __name__ == "__main__":
    run_few_shot_benchmark()