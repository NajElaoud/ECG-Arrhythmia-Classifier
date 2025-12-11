"""
ECG AI Project - Complete Code with Proper Class Balancing
Dataset: PTB-XL (automatically downloaded)
Models: CNN, LSTM, and hybrid architectures included (using CUDA or CPU)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    STRATIFIED_AVAILABLE = True
except ImportError:
    print("Warning: iterative-stratification not installed. Using regular split.")
    print("Install with: pip install iterative-stratification")
    STRATIFIED_AVAILABLE = False

from ecg_visualization import visualize_model_results

# ============================================
# STEP 1: Download and Load PTB-XL Dataset
# ============================================

def download_ptbxl(data_path='./ptbxl/'):
    """
    Download PTB-XL dataset from PhysioNet
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("Downloading PTB-XL dataset...")
        print("Please run these commands in terminal:")
        print(f"wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P {data_path}")
        print("\nOr use:")
        print("pip install wfdb")
        print("Then download from: https://physionet.org/content/ptb-xl/1.0.3/")
    return data_path

def load_ptbxl_data(data_path='./ptbxl/', sampling_rate=100):
    """
    Load PTB-XL dataset
    Args:
        data_path: path to PTB-XL directory
        sampling_rate: 100 or 500 Hz
    Returns:
        X: ECG signals, Y: labels, metadata
    """
    # Load metadata
    metadata_file = os.path.join(data_path, 'ptbxl_database.csv')
    Y = pd.read_csv(metadata_file, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: eval(x))
    
    # Load signals
    if sampling_rate == 100:
        data_folder = os.path.join(data_path, 'records100')
    else:
        data_folder = os.path.join(data_path, 'records500')
    
    X = []
    valid_indices = []
    
    print("Loading ECG signals...")
    for idx, row in Y.iterrows():
        file_path = os.path.join(data_path, row.filename_lr if sampling_rate == 100 else row.filename_hr)
        try:
            signal, _ = wfdb.rdsamp(file_path)
            X.append(signal)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    X = np.array(X)
    Y = Y.loc[valid_indices]
    
    print(f"Loaded {len(X)} ECG records")
    print(f"Signal shape: {X.shape}")  # (num_samples, time_steps, num_leads)
    
    return X, Y

# ============================================
# STEP 2: Data Preprocessing with Class Filtering
# ============================================

def preprocess_labels(Y, data_path='./ptbxl/', target_classes=['NORM', 'MI', 'STTC', 'CD', 'HYP'], min_samples=50):
    """
    Extract and process diagnostic labels with automatic filtering
    Args:
        Y: metadata dataframe
        data_path: path to PTB-XL directory (for scp_statements.csv)
        target_classes: list of target diagnostic superclasses
        min_samples: minimum number of samples required per class
    Returns:
        Binary labels for classification, filtered class names, MultiLabelBinarizer
    """
    # Load SCP statements to get superclass mappings
    scp_statements_file = os.path.join(data_path, 'scp_statements.csv')
    if os.path.exists(scp_statements_file):
        scp_statements = pd.read_csv(scp_statements_file, index_col=0)
        scp_statements = scp_statements[scp_statements.diagnostic == 1]
        
        def aggregate_diagnostic(scp_codes):
            """Aggregate SCP codes into diagnostic superclasses"""
            labels = []
            for key in scp_codes.keys():
                if key in scp_statements.index:
                    superclass = scp_statements.loc[key].diagnostic_class
                    if superclass in target_classes:
                        labels.append(superclass)
            return list(set(labels))  # Remove duplicates
        
        Y['diagnostic_labels'] = Y.scp_codes.apply(aggregate_diagnostic)
    else:
        print(f"Warning: {scp_statements_file} not found. Using direct SCP code matching.")
        def extract_labels(scp_codes, target_classes):
            labels = []
            for tc in target_classes:
                if tc in scp_codes:
                    labels.append(tc)
            return labels
        
        Y['diagnostic_labels'] = Y.scp_codes.apply(lambda x: extract_labels(x, target_classes))
    
    # Convert to binary format
    mlb = MultiLabelBinarizer(classes=target_classes)
    y_binary = mlb.fit_transform(Y['diagnostic_labels'])
    
    print(f"\n=== Initial Label Distribution ===")
    for i, label in enumerate(target_classes):
        count = y_binary[:, i].sum()
        percentage = (count / len(y_binary)) * 100
        print(f"{label}: {count} samples ({percentage:.2f}%)")
    
    # Filter out classes with too few samples
    valid_classes = []
    valid_indices = []
    
    for i, label in enumerate(target_classes):
        count = y_binary[:, i].sum()
        if count >= min_samples:
            valid_classes.append(label)
            valid_indices.append(i)
        else:
            print(f"⚠ Warning: {label} has only {count} samples (< {min_samples}). Excluding from training.")
    
    if len(valid_classes) == 0:
        raise ValueError(f"No classes have at least {min_samples} samples. Try lowering min_samples parameter.")
    
    # Keep only valid classes
    y_binary_filtered = y_binary[:, valid_indices]
    
    print(f"\n=== Final Classes Used ({len(valid_classes)} classes) ===")
    for i, label in enumerate(valid_classes):
        count = y_binary_filtered[:, i].sum()
        percentage = (count / len(y_binary_filtered)) * 100
        print(f"{label}: {count} samples ({percentage:.2f}%)")
    
    return y_binary_filtered, valid_classes, mlb

def normalize_signals(X_train, X_val, X_test):
    """
    Normalize ECG signals
    """
    scaler = StandardScaler()
    
    # Reshape for scaling
    n_train, n_steps, n_leads = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_leads)
    
    # Fit on training data
    scaler.fit(X_train_reshaped)
    
    # Transform all sets
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_leads)).reshape(n_train, n_steps, n_leads)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_leads)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_leads)).reshape(X_test.shape)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ============================================
# STEP 3: PyTorch Dataset Class
# ============================================

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

# ============================================
# STEP 4: Model Architectures
# ============================================

class CNN_ECG(nn.Module):
    """
    1D CNN for ECG classification
    """
    def __init__(self, num_classes=5, num_leads=12):
        super(CNN_ECG, self).__init__()
        
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, time_steps, leads)
        x = x.permute(0, 2, 1)  # -> (batch, leads, time_steps)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.global_pool(x).squeeze(-1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)

class LSTM_ECG(nn.Module):
    """
    LSTM for ECG classification
    """
    def __init__(self, num_classes=5, num_leads=12, hidden_size=128):
        super(LSTM_ECG, self).__init__()
        
        self.lstm1 = nn.LSTM(num_leads, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, batch_first=True, bidirectional=True)
        
        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, time_steps, leads)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Take last time step
        x = x[:, -1, :]
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)

class Hybrid_CNN_LSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture
    """
    def __init__(self, num_classes=5, num_leads=12):
        super(Hybrid_CNN_LSTM, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        
        # Classifier
        self.fc1 = nn.Linear(128*2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, time_steps, leads)
        x = x.permute(0, 2, 1)  # -> (batch, leads, time_steps)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = x.permute(0, 2, 1)  # -> (batch, time_steps, features)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last time step
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)

# ============================================
# STEP 5: Training Functions
# ============================================

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda', model_name='best_ecg_model'):
    """
    Train the model
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Ensure model directory exists and save
            model_dir = os.path.join('results', 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, target_classes, device='cuda'):
    """
    Evaluate model performance
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Convert probabilities to binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, binary_preds, 
                                target_names=target_classes,
                                zero_division=0))
    
    # Calculate AUC-ROC for each class
    print("\n=== AUC-ROC Scores ===")
    for i, label in enumerate(target_classes):
        try:
            # Check if we have both positive and negative samples
            if len(np.unique(all_labels[:, i])) > 1:
                auc_score = roc_auc_score(all_labels[:, i], all_preds[:, i])
                print(f"{label}: {auc_score:.4f}")
            else:
                print(f"{label}: N/A (only one class present in test set)")
        except Exception as e:
            print(f"{label}: N/A (error: {str(e)})")
    
    return all_preds, all_labels, binary_preds

# ============================================
# STEP 6: Visualization Functions
# ============================================

def plot_ecg_sample(signal, title="ECG Signal"):
    """
    Plot a single ECG recording (all 12 leads)
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(12, 1, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    for i in range(12):
        axes[i].plot(signal[:, i], linewidth=0.5)
        axes[i].set_ylabel(lead_names[i])
        axes[i].grid(True, alpha=0.3)
        if i < 11:
            axes[i].set_xticks([])
    
    axes[-1].set_xlabel('Time (samples)')
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, val_losses):
    """
    Plot training history
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix for each class
    """
    num_classes = len(class_names)
    rows = (num_classes + 2) // 3
    cols = min(3, num_classes)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if num_classes == 1:
        axes = [axes]
    else:
        axes = axes.ravel() if num_classes > 1 else [axes]
    
    for i, class_name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{class_name}')
        axes[i].set_ylabel('True')
        axes[i].set_xlabel('Predicted')
    
    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================
# STEP 7: Stratified Splitting
# ============================================

def stratified_split(X, y, test_size=0.3, random_state=42):
    """
    Perform stratified split for multi-label data
    """
    if STRATIFIED_AVAILABLE and y.shape[1] > 1:
        print("Using stratified split for multi-label data...")
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        
        for train_idx, test_idx in msss.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
        return X_train, X_test, y_train, y_test
    else:
        print("Using regular stratified split...")
        # For single-label or when stratified library not available
        # Use the dominant class for stratification
        stratify_labels = y.argmax(axis=1) if y.shape[1] > 1 else y.ravel()
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels)

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Set device with GPU optimization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠ WARNING: CUDA not available. Training will be slow on CPU.")
        print("Install CUDA-enabled PyTorch with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # Configuration
    DATA_PATH = './ptbxl/'
    SAMPLING_RATE = 100  # Hz
    BATCH_SIZE = 32  # Increase if your GPU has enough memory
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    MIN_SAMPLES_PER_CLASS = 50  # Minimum samples required per class
    
    # Step 1: Load data
    print("\n" + "="*60)
    print("STEP 1: LOADING PTB-XL DATASET")
    print("="*60)
    download_ptbxl(DATA_PATH)
    X, Y = load_ptbxl_data(DATA_PATH, SAMPLING_RATE)
    
    # Step 2: Preprocess labels with automatic filtering
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING LABELS")
    print("="*60)
    initial_target_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y_binary, target_classes, mlb = preprocess_labels(Y, DATA_PATH, initial_target_classes, MIN_SAMPLES_PER_CLASS)
    
    if len(target_classes) == 0:
        print("\n⚠ ERROR: No classes with sufficient samples!")
        print(f"Try lowering MIN_SAMPLES_PER_CLASS (currently {MIN_SAMPLES_PER_CLASS})")
        exit(1)
    
    # Step 3: Train-Val-Test split with stratification
    print("\n" + "="*60)
    print("STEP 3: SPLITTING DATA")
    print("="*60)
    
    # First split: train vs temp (val+test)
    X_train, X_temp, y_train, y_temp = stratified_split(X, y_binary, test_size=0.3, random_state=42)
    
    # Second split: val vs test
    X_val, X_test, y_val, y_test = stratified_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Verify class distribution in each split
    print("\n=== Class Distribution in Splits ===")
    for i, class_name in enumerate(target_classes):
        train_count = y_train[:, i].sum()
        val_count = y_val[:, i].sum()
        test_count = y_test[:, i].sum()
        print(f"{class_name}: Train={train_count}, Val={val_count}, Test={test_count}")
    
    # Step 4: Normalize
    print("\n" + "="*60)
    print("STEP 4: NORMALIZING SIGNALS")
    print("="*60)
    X_train, X_val, X_test, scaler = normalize_signals(X_train, X_val, X_test)
    
    # Step 5: Create DataLoaders
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Step 6: Visualize sample
    print("\n" + "="*60)
    print("STEP 5: VISUALIZING SAMPLE ECG")
    print("="*60)
    plot_ecg_sample(X_train[0], "Sample ECG - 12 Leads")
    
    # Step 7: Initialize and train model
    print("\n" + "="*60)
    print("STEP 6: MODEL TRAINING")
    print("="*60)
    print("Choose model: 1=CNN, 2=LSTM, 3=Hybrid")
    model_choice = 2  # Change this to select different models
    
    if model_choice == 1:
        model = CNN_ECG(num_classes=len(target_classes), num_leads=12).to(device)
        model_type = 'cnn'
        print("Using CNN model")
    elif model_choice == 2:
        model = LSTM_ECG(num_classes=len(target_classes), num_leads=12).to(device)
        model_type = 'lstm'
        print("Using LSTM model")
    else:
        model = Hybrid_CNN_LSTM(num_classes=len(target_classes), num_leads=12).to(device)
        model_type = 'hybrid'
        print("Using Hybrid CNN-LSTM model")
    
    # Check if model already exists
    SKIP_TRAINING = False  # Set to True to skip training and load existing model
    # model path under results/models (per model type)
    model_dir = os.path.join('results', 'models')
    model_name = f"best_ecg_model_{model_type}"
    model_path = os.path.join(model_dir, f"{model_name}.pth")

    if SKIP_TRAINING and os.path.exists(model_path):
        print("\n=== Loading Pre-trained Model ===")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
        train_losses = []
        val_losses = []
    else:
        print("\n=== Training Model ===")
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device,
            model_name=model_name
        )
        
        # Clear GPU cache after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Plot training history
        plot_training_history(train_losses, val_losses)
    
    # Step 8: Evaluate on test set
    print("\n" + "="*60)
    print("STEP 7: EVALUATING ON TEST SET")
    print("="*60)
    if not SKIP_TRAINING or not os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    predictions, labels, binary_preds = evaluate_model(model, test_loader, target_classes, device)
    
    # Step 9: Plot confusion matrices
    plot_confusion_matrix(labels, binary_preds, target_classes)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved as: {model_path}")
    print(f"Number of classes: {len(target_classes)}")
    print(f"Classes: {', '.join(target_classes)}")
    
    # Step 10: Generate all visualizations
    print("\n" + "="*60)
    print("STEP 8: GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    visualize_model_results(
        model=model,
        test_loader=test_loader,
        device=device,
        num_detailed=2,  # Number of detailed dashboards
        output_dir=os.path.join('results', 'visualizations')
    )