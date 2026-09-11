import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from src.data_loader import load_dataset

def run_zero_shot_loso():
    test_subjects = list(range(1, 61))
    
    print("[1/2] Loading 60 subjects with Per-Subject ESA (Unsupervised)...", flush=True)
    # use_esa=True applies R_sub^(-1/2) per subject before cross-validation
    X, y, groups = load_dataset(
        subject_ids=test_subjects, 
        runs=[4, 8, 12], 
        motor_only=True, 
        use_esa=True,
        use_sliding_window=False
    )

    print("[2/2] Projecting to Tangent Space and Running 60-Fold LOSO...", flush=True)
    cov_estimator = Covariances(estimator='lwf')
    C_all = cov_estimator.fit_transform(X)

    # Tangent Space projection centered at Identity (since ESA pre-aligned all subjects to I)
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

    print(f"\n--> True Zero-Shot Mean LOSO Accuracy (60 Subjects): {np.mean(scores) * 100:.1f}%")

if __name__ == "__main__":
    run_zero_shot_loso()