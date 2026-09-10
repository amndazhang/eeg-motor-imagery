import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from src.data_loader import load_dataset
from src.models import create_riemannian_pipeline

def evaluate_loso_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    logo = LeaveOneGroupOut()
    scores = {}
    
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        test_sub = groups[test_idx][0]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        pipeline = create_riemannian_pipeline()
        pipeline.fit(X_train, y_train)
        acc = pipeline.score(X_test, y_test)
        scores[test_sub] = float(acc)
        
    return scores

if __name__ == "__main__":
    test_subjects = list(range(1, 61))
    print("Loading 60 subjects with ESA and Sliding Window Augmentation...\n")
    
    X, y, groups = load_dataset(
        subject_ids=test_subjects, 
        runs=[4, 8, 12], 
        motor_only=True, 
        use_esa=True,
        use_sliding_window=True
    )
    
    print(f"Dataset expanded: {X.shape[0]} windows, {X.shape[1]} channels, {X.shape[2]} timepoints per window.")
    
    print("\n--- Evaluating Riemannian Tangent Space + Sliding Window (LOSO 60 Subjects) ---")
    riemann_scores = evaluate_loso_model(X, y, groups)
    print(f"\nMean Riemannian Augment LOSO Accuracy (60 Subjects): {np.mean(list(riemann_scores.values())) * 100:.1f}%")