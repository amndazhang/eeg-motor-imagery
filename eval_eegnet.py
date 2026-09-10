import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from src.data_loader import load_dataset
from src.eegnet import EEGNet

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon acceleration
    return torch.device("cpu")

def zscore_normalize_trials(X: np.ndarray) -> np.ndarray:
    """
    Normalizes each trial per channel (zero mean, unit variance).
    Prevents gradient explosion from raw microvolt EEG inputs.
    """
    means = X.mean(axis=-1, keepdims=True)
    stds = X.std(axis=-1, keepdims=True) + 1e-8
    return (X - means) / stds

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
    return total_loss / total, correct / total

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
    return correct / total

def run_eegnet_loso():
    device = get_device()
    print(f"Using compute acceleration device: {device}", flush=True)
    
    test_subjects = list(range(1, 61))
    print("Loading raw trial data for 60 subjects...", flush=True)
    
    X_raw, y_raw, groups = load_dataset(
        subject_ids=test_subjects, 
        runs=[4, 8, 12], 
        motor_only=True, 
        use_esa=False,
        use_sliding_window=False
    )
    
    # Per-trial channel z-scoring
    X_norm = zscore_normalize_trials(X_raw)
    
    # Reshape for 2D Conv PyTorch format: (N_trials, 1, channels, samples)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_raw, dtype=torch.long)
    
    logo = LeaveOneGroupOut()
    scores = []
    
    print("\n--- Starting EEGNet Cross-Subject (LOSO 60 Subjects) Benchmark ---", flush=True)
    
    epochs = 25
    batch_size = 64
    lr = 0.001
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X_norm, y_raw, groups=groups)):
        # Train / Test splitting
        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_test, y_test = X_tensor[test_idx], y_tensor[test_idx]
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        
        # Instantiate fresh model for each Leave-One-Subject-Out fold
        model = EEGNet(n_classes=2, channels=X_norm.shape[1], samples=X_norm.shape[2]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        for epoch in range(epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            
        acc = evaluate(model, test_loader, device)
        scores.append(acc)
        
        if (fold + 1) % 10 == 0:
            print(f"  Processed {fold + 1}/60 subjects | Running Mean Accuracy: {np.mean(scores) * 100:.1f}%", flush=True)
            
    print(f"\n--> Final Mean EEGNet Zero-Shot LOSO Accuracy (60 Subjects): {np.mean(scores) * 100:.1f}%")

if __name__ == "__main__":
    run_eegnet_loso()