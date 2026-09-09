import numpy as np
from typing import Tuple, List
from src.preprocessing import preprocess_subject

def load_dataset(
    subject_ids: list, 
    runs: list = [4, 8, 12],
    motor_only: bool = False
):
    X_list, y_list, groups_list = [], [], []

    for sub_id in subject_ids:
        try:
            epochs = preprocess_subject(subject_id=sub_id, runs=runs, motor_only=motor_only)
            X_sub = epochs.get_data(copy=True)
            raw_labels = epochs.events[:, -1]
            t1_val = epochs.event_id['T1']
            y_sub = np.where(raw_labels == t1_val, 0, 1)
            
            X_list.append(X_sub)
            y_list.append(y_sub)
            groups_list.extend([sub_id] * len(y_sub))
        except Exception as e:
            print(f"Warning: Failed to load subject {sub_id} — {e}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.array(groups_list)

    return X, y, groups