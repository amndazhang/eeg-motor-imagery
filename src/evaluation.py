import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from src.models import create_csp_lda_pipeline

def evaluate_within_subject(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """
    Evaluates 5-fold CV accuracy per subject independently.
    """
    unique_subs = np.unique(groups)
    subject_accuracies = {}
    
    for sub in unique_subs:
        mask = (groups == sub)
        X_sub, y_sub = X[mask], y[mask]
        
        pipeline = create_csp_lda_pipeline()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in skf.split(X_sub, y_sub):
            pipeline.fit(X_sub[train_idx], y_sub[train_idx])
            scores.append(pipeline.score(X_sub[test_idx], y_sub[test_idx]))
            
        subject_accuracies[sub] = float(np.mean(scores))
        
    return subject_accuracies

def evaluate_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """
    Evaluates Leave-One-Subject-Out (LOSO) zero-shot cross-subject accuracy.
    """
    logo = LeaveOneGroupOut()
    loso_scores = {}
    
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        test_sub = groups[test_idx][0]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        pipeline = create_csp_lda_pipeline()
        pipeline.fit(X_train, y_train)
        acc = pipeline.score(X_test, y_test)
        loso_scores[test_sub] = float(acc)
        
    return loso_scores

def evaluate_execution_to_imagery_transfer(X_exec: np.ndarray, y_exec: np.ndarray, 
                                            X_imag: np.ndarray, y_imag: np.ndarray, 
                                            groups: np.ndarray) -> dict:
    """
    Trains CSP+LDA on Motor Execution data and evaluates on Motor Imagery data (LOSO scheme).
    """
    logo = LeaveOneGroupOut()
    transfer_scores = {}
    
    for train_idx, test_idx in logo.split(X_exec, y_exec, groups=groups):
        test_sub = groups[test_idx][0]
        
        # Train ONLY on Execution runs of (N-1) subjects
        X_train, y_train = X_exec[train_idx], y_exec[train_idx]
        
        # Test ONLY on Imagery runs of the held-out subject
        imag_mask = (groups == test_sub)
        X_test, y_test = X_imag[imag_mask], y_imag[imag_mask]
        
        pipeline = create_csp_lda_pipeline()
        pipeline.fit(X_train, y_train)
        acc = pipeline.score(X_test, y_test)
        transfer_scores[test_sub] = float(acc)
        
    return transfer_scores