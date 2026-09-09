import numpy as np
from src.data_loader import load_dataset
from src.evaluation import evaluate_loso, evaluate_execution_to_imagery_transfer

if __name__ == "__main__":
    test_subjects = list(range(1, 11))
    print(f"Loading data for subjects: {test_subjects}...\n")
    
    # 1. Load Motor Imagery (Runs 4, 8, 12)
    print("Loading Motor Imagery runs [4, 8, 12]...")
    X_imag, y_imag, groups = load_dataset(subject_ids=test_subjects, runs=[4, 8, 12])
    
    # 2. Load Motor Execution (Runs 3, 7, 11)
    print("Loading Motor Execution runs [3, 7, 11]...")
    X_exec, y_exec, _ = load_dataset(subject_ids=test_subjects, runs=[3, 7, 11])
    
    # 3. Evaluate Pure Imagery LOSO
    print("\n--- Pure Imagery Zero-Shot (LOSO) ---")
    imag_loso_accs = evaluate_loso(X_imag, y_imag, groups)
    print(f"Mean Pure Imagery LOSO Accuracy: {np.mean(list(imag_loso_accs.values())) * 100:.1f}%")
    
    # 4. Evaluate Execution -> Imagery Transfer
    print("\n--- Execution -> Imagery Cross-Condition Transfer (LOSO) ---")
    transfer_accs = evaluate_execution_to_imagery_transfer(X_exec, y_exec, X_imag, y_imag, groups)
    for sub, acc in transfer_accs.items():
        print(f"Subject {sub:02d} Transfer Accuracy: {acc * 100:.1f}%")
        
    print(f"\nMean Execution -> Imagery Transfer Accuracy: {np.mean(list(transfer_accs.values())) * 100:.1f}%")

    # Run comparison with motor-only channels
    print("\nLoading Motor Imagery (21 Motor Channels Only)...")
    X_imag_m, y_imag_m, groups_m = load_dataset(subject_ids=test_subjects, runs=[4, 8, 12], motor_only=True)
    
    print("--- Motor-Only Channels Pure Imagery Zero-Shot (LOSO) ---")
    imag_m_accs = evaluate_loso(X_imag_m, y_imag_m, groups_m)
    print(f"Mean Motor-Only Imagery LOSO Accuracy: {np.mean(list(imag_m_accs.values())) * 100:.1f}%")