from pathlib import Path
import joblib
from src.data_loader import load_dataset
from src.models import create_riemannian_pipeline

if __name__ == "__main__":
    # Scaled from 20 to 60 subjects
    training_subjects = list(range(1, 61))
    print(f"Training production Riemannian model on subjects 1-60...")
    
    X_train, y_train, _ = load_dataset(
        subject_ids=training_subjects, 
        runs=[3, 7, 11], 
        motor_only=True, 
        use_esa=True
    )
    
    pipeline = create_riemannian_pipeline()
    pipeline.fit(X_train, y_train)
    
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/global_riemann_model.joblib")
    print("Successfully trained and saved 60-subject Riemannian model to models/global_riemann_model.joblib")