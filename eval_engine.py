import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from src.data_loader import load_dataset
from src.models import create_riemannian_pipeline

def run_scheme_1_within_subject(subjects):
    """Scheme 1: Within-Subject 5-Fold Stratified CV (Performance Ceiling)"""
    print("\n--- Running Scheme 1: Within-Subject CV (Performance Ceiling) ---", flush=True)
    
    X, y, groups = load_dataset(
        subject_ids=subjects, 
        runs=[4, 8, 12], 
        motor_only=True, 
        use_esa=False, 
        use_sliding_window=False
    )
    
    subject_scores = []
    loaded_subs = np.unique(groups)
    
    for sub in loaded_subs:
        sub_mask = (groups == sub)
        X_sub, y_sub = X[sub_mask], y[sub_mask]
        
        if len(X_sub) < 5 or len(np.unique(y_sub)) < 2:
            continue
            
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        for train_idx, test_idx in skf.split(X_sub, y_sub):
            pipeline = create_riemannian_pipeline()
            pipeline.fit(X_sub[train_idx], y_sub[train_idx])
            acc = pipeline.score(X_sub[test_idx], y_sub[test_idx])
            fold_scores.append(acc)
            
        subject_scores.append(np.mean(fold_scores))
        
    mean_acc = np.mean(subject_scores)
    std_acc = np.std(subject_scores)
    print(f"--> Scheme 1 Accuracy (Within-Subject): {mean_acc * 100:.1f}% ± {std_acc * 100:.1f}%")
    return mean_acc, std_acc


def run_scheme_2_loso(subjects):
    """Scheme 2: Leave-One-Subject-Out Cross-Validation (Zero-Shot Generalization)"""
    print("\n--- Running Scheme 2: LOSO Cross-Subject CV (Zero-Shot Generalization) ---", flush=True)
    
    X, y, groups = load_dataset(
        subject_ids=subjects, 
        runs=[4, 8, 12], 
        motor_only=True, 
        use_esa=True, 
        use_sliding_window=False
    )
    
    cov_estimator = Covariances(estimator='lwf')
    C_all = cov_estimator.fit_transform(X)
    
    ts = TangentSpace(metric='riemann')
    T_all = ts.fit_transform(C_all)
    
    logo = LeaveOneGroupOut()
    scores = []
    
    for train_idx, test_idx in logo.split(T_all, y, groups=groups):
        T_train, y_train = T_all[train_idx], y[train_idx]
        T_test, y_test = T_all[test_idx], y[test_idx]
        
        clf = LogisticRegression(C=0.1, solver='lbfgs', max_iter=500)
        clf.fit(T_train, y_train)
        scores.append(clf.score(T_test, y_test))
        
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"--> Scheme 2 Accuracy (Zero-Shot LOSO): {mean_acc * 100:.1f}% ± {std_acc * 100:.1f}%")
    return mean_acc, std_acc


def run_scheme_3_execution_to_imagery(subjects):
    """Scheme 3: Motor Execution (Runs 3,7,11) to Motor Imagery (Runs 4,8,12) Transfer"""
    print("\n--- Running Scheme 3: Execution-to-Imagery Cross-Task Transfer ---", flush=True)
    
    X_exec, y_exec, groups_exec = load_dataset(
        subject_ids=subjects, 
        runs=[3, 7, 11], 
        motor_only=True, 
        use_esa=True, 
        use_sliding_window=False
    )
    
    X_imag, y_imag, groups_imag = load_dataset(
        subject_ids=subjects, 
        runs=[4, 8, 12], 
        motor_only=True, 
        use_esa=True, 
        use_sliding_window=False
    )
    
    # Intersect to evaluate subjects present in both sets
    valid_subs = np.intersect1d(np.unique(groups_exec), np.unique(groups_imag))
    
    within_scores = []
    for sub in valid_subs:
        exec_mask = (groups_exec == sub)
        imag_mask = (groups_imag == sub)
        
        X_tr, y_tr = X_exec[exec_mask], y_exec[exec_mask]
        X_te, y_te = X_imag[imag_mask], y_imag[imag_mask]
        
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
            
        pipeline = create_riemannian_pipeline()
        pipeline.fit(X_tr, y_tr)
        within_scores.append(pipeline.score(X_te, y_te))
        
    within_mean = np.mean(within_scores) if len(within_scores) > 0 else 0.0
    within_std = np.std(within_scores) if len(within_scores) > 0 else 0.0
    
    # Cross-Subject Execution-to-Imagery Transfer
    exec_valid_mask = np.isin(groups_exec, valid_subs)
    imag_valid_mask = np.isin(groups_imag, valid_subs)
    
    X_exec_v, y_exec_v, groups_exec_v = X_exec[exec_valid_mask], y_exec[exec_valid_mask], groups_exec[exec_valid_mask]
    X_imag_v, y_imag_v, groups_imag_v = X_imag[imag_valid_mask], y_imag[imag_valid_mask], groups_imag[imag_valid_mask]
    
    cov_estimator = Covariances(estimator='lwf')
    ts = TangentSpace(metric='riemann')
    
    C_exec = cov_estimator.fit_transform(X_exec_v)
    C_imag = cov_estimator.fit_transform(X_imag_v)
    
    C_combined = np.concatenate([C_exec, C_imag], axis=0)
    ts.fit(C_combined)
    
    T_exec = ts.transform(C_exec)
    T_imag = ts.transform(C_imag)
    
    logo = LeaveOneGroupOut()
    loso_scores = []
    
    for train_idx, test_idx in logo.split(T_exec, y_exec_v, groups=groups_exec_v):
        test_sub = groups_exec_v[test_idx[0]]
        test_imag_idx = np.where(groups_imag_v == test_sub)[0]
        
        if len(test_imag_idx) == 0:
            continue
            
        T_train, y_train = T_exec[train_idx], y_exec_v[train_idx]
        T_test, y_test = T_imag[test_imag_idx], y_imag_v[test_imag_idx]
        
        clf = LogisticRegression(C=0.1, solver='lbfgs', max_iter=500)
        clf.fit(T_train, y_train)
        loso_scores.append(clf.score(T_test, y_test))
        
    loso_mean = np.mean(loso_scores) if len(loso_scores) > 0 else 0.0
    loso_std = np.std(loso_scores) if len(loso_scores) > 0 else 0.0
    
    print(f"--> Scheme 3a (Within-Subject Execution->Imagery Transfer): {within_mean * 100:.1f}% ± {within_std * 100:.1f}%")
    print(f"--> Scheme 3b (Cross-Subject Execution->Imagery Transfer): {loso_mean * 100:.1f}% ± {loso_std * 100:.1f}%")
    
    return within_mean, loso_mean


def main():
    test_subjects = list(range(1, 61))
    print("=================================================================")
    print("        COMPREHENSIVE BCI EVALUATION ENGINE BENCHMARK            ")
    print("=================================================================")
    
    s1_mean, s1_std = run_scheme_1_within_subject(test_subjects)
    s2_mean, s2_std = run_scheme_2_loso(test_subjects)
    s3_within, s3_loso = run_scheme_3_execution_to_imagery(test_subjects)
    
    print("\n=================================================================")
    print("                     FINAL RESULTS SUMMARY                       ")
    print("=================================================================")
    print(f"Scheme 1: Within-Subject CV (Performance Ceiling)  : {s1_mean * 100:.1f}%")
    print(f"Scheme 2: LOSO Cross-Subject (Zero-Shot Imagery)   : {s2_mean * 100:.1f}%")
    print(f"Scheme 3a: Execution-to-Imagery (Within-Subject)   : {s3_within * 100:.1f}%")
    print(f"Scheme 3b: Execution-to-Imagery (Cross-Subject LOSO): {s3_loso * 100:.1f}%")
    print("=================================================================")

if __name__ == "__main__":
    main()