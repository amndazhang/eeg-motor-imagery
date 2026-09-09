from pathlib import Path
import joblib
from src.data_loader import load_dataset
from src.models import create_csp_lda_pipeline

if __name__ == "__main__":
    training_subjects = list(range(1, 21))
    print(f"Training global CSP+LDA model on subjects: {training_subjects}...")
    
    # Train on Motor Execution runs for max SNR
    X_train, y_train, _ = load_dataset(subject_ids=training_subjects, runs=[3, 7, 11], motor_only=True)
    
    pipeline = create_csp_lda_pipeline(n_components=4)
    pipeline.fit(X_train, y_train)
    
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/global_csp_lda.joblib")
    print("Successfully trained and saved global model to models/global_csp_lda.joblib")