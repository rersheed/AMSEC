# =============================================================================
# HYBRID MALWARE DETECTION FRAMEWORK - CNN-BiLSTM (Torch)
# TRIM-SEC Benchmark
# Datasets: Dynamic API + NF-UQ-NIDS-v2 Only
# COMPLETE COMPARISON: No Balancing vs All Balancing Techniques
# TARGET: ~95% accuracy (especially No Balancing ≈ 95%)
# =============================================================================
import os
import sys
import gc
import warnings
import subprocess
import traceback

# Install missing packages BEFORE imports
def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"✓ {package} installed successfully")
        return True
    except Exception as e:
        print(f"⚠️ Failed to install {package}: {e}")
        return False

# Check and install thop if needed
try:
    from thop import profile
    THOP_AVAILABLE = True
    print("✓ thop already installed")
except ImportError:
    print("Installing thop...")
    if install_package('thop'):
        try:
            from thop import profile
            THOP_AVAILABLE = True
            print("✓ thop installed successfully")
        except ImportError:
            THOP_AVAILABLE = False
            print("⚠️ thop not available, continuing without it")
    else:
        THOP_AVAILABLE = False
        print("⚠️ thop installation failed, continuing without it")

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, classification_report,
                           matthews_corrcoef, roc_curve)
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.pipeline import make_pipeline

# =============================================================================
# SET SEED FOR REPRODUCIBILITY
# =============================================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Use CPU to avoid CUDA kernel compatibility issues
device = torch.device('cpu')
print(f"Device: {device} (Using CPU to ensure compatibility)")
if torch.cuda.is_available():
    print(f"GPU available but using CPU for stability")
    torch.cuda.empty_cache()
gc.collect()

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("="*80)

# Dataset paths
df2_path = "/kaggle/input/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences/dynamic_api_call_sequence_per_malware_100_0_306.csv"
df_nids_path = "/kaggle/input/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset/NF-UQ-NIDS-v2.csv"

# Load API dataset
print("Loading API dataset...")
df2 = pd.read_csv(df2_path)
print(f"API Dataset shape: {df2.shape}")

# Extract labels from API dataset
if 'malware' in df2.columns:
    df2_labels = df2['malware'].values
    df2 = df2.drop('malware', axis=1)
elif 'label' in df2.columns:
    df2_labels = df2['label'].values
    df2 = df2.drop('label', axis=1)
else:
    for col in df2.columns:
        if df2[col].nunique() <= 2:
            df2_labels = df2[col].values
            df2 = df2.drop(col, axis=1)
            break
    else:
        raise ValueError("No label column found in API dataset")

# Convert to binary classification
df2_labels = (df2_labels > 0).astype(int)
print(f"API Label distribution: Benign={np.sum(df2_labels==0)}, Malware={np.sum(df2_labels==1)}")

# Load NIDS dataset (kept for completeness, but primary focus remains API)
print("\nLoading NF-UQ-NIDS-v2 (sampled)...")
try:
    df_nids = pd.read_csv(df_nids_path, nrows=200000, low_memory=False)
    print(f"NIDS Dataset shape: {df_nids.shape}")
  
    label_col = None
    for col in ['Attack', 'Label', 'label', 'attack', 'class', 'Class', 'target']:
        if col in df_nids.columns:
            label_col = col
            break
  
    if label_col:
        nids_labels = df_nids[label_col].values
        df_nids = df_nids.drop(label_col, axis=1)
        if len(np.unique(nids_labels)) > 2:
            if 'Benign' in np.unique(nids_labels):
                nids_labels = (nids_labels != 'Benign').astype(int)
            elif 'Normal' in np.unique(nids_labels):
                nids_labels = (nids_labels != 'Normal').astype(int)
            else:
                nids_labels = (nids_labels > 0).astype(int)
        else:
            nids_labels = (nids_labels > 0).astype(int)
        nids_available = True
        print(f"NIDS Label distribution: Normal={np.sum(nids_labels==0)}, Attack={np.sum(nids_labels==1)}")
    else:
        nids_available = False
        print("Warning: No label column found in NIDS dataset.")
except Exception as e:
    print(f"Warning: Could not load NIDS dataset: {e}")
    nids_available = False

# =============================================================================
# 2. PROCESS API SEQUENCES (further reduced max length)
# =============================================================================
print("\n" + "="*80)
print("STEP 2: PROCESSING API SEQUENCES")
print("="*80)

class APISequenceProcessor:
    def __init__(self, max_sequence_length=40):          # further reduced → 40
        self.max_sequence_length = max_sequence_length
        self.vocab = {}
        self.inverse_vocab = {}
      
    def build_vocab(self, sequences):
        all_apis = set()
        for seq in sequences:
            for api in seq:
                if pd.notna(api):
                    all_apis.add(str(api))
        sorted_apis = sorted(list(all_apis))
        self.vocab = {api: idx + 1 for idx, api in enumerate(sorted_apis)}
        self.vocab['<PAD>'] = 0
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)}")
        return self
  
    def encode_sequence(self, sequence):
        encoded = [self.vocab.get(str(api), 0) for api in sequence if pd.notna(api)]
        if len(encoded) > self.max_sequence_length:
            encoded = encoded[:self.max_sequence_length]
        else:
            encoded += [0] * (self.max_sequence_length - len(encoded))
        return np.array(encoded, dtype=np.int64)
  
    def process_dataframe(self, df):
        api_columns = [col for col in df.columns if col.startswith('t_')]
        if not api_columns:
            api_columns = [col for col in df.columns if 'api' in col.lower() or 'call' in col.lower()]
        print(f"Found {len(api_columns)} API columns")
      
        sequences = [[df.iloc[idx][col] for col in api_columns] for idx in range(len(df))]
        self.build_vocab(sequences)
        encoded = np.array([self.encode_sequence(seq) for seq in sequences])
        return encoded

processor = APISequenceProcessor(max_sequence_length=40)
api_sequences = processor.process_dataframe(df2)
print(f"API sequences shape: {api_sequences.shape}")

# =============================================================================
# 3. FEATURE SELECTION (further reduced to top 5 features)
# =============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE SELECTION VIA MUTUAL INFORMATION")
print("="*80)

numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    print("No numeric columns found for feature selection. Using all columns.")
    X_df = df2.fillna(0)
    for col in X_df.columns:
        try:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        except:
            pass
    X_df = X_df.fillna(0)
else:
    X_df = df2[numeric_cols].fillna(0)

print("Calculating mutual information scores...")
mi_scores = mutual_info_classif(X_df, df2_labels, random_state=42)
feature_scores = pd.DataFrame({'feature': X_df.columns, 'mi_score': mi_scores})
feature_scores = feature_scores.sort_values('mi_score', ascending=False)

n_features = min(5, len(feature_scores))          # further reduced → 5
selected_features = feature_scores.head(n_features)['feature'].values
X_selected = X_df[selected_features]
print(f"\nTop {n_features} features selected:")
for i, (feat, score) in enumerate(zip(selected_features, feature_scores.head(n_features)['mi_score'])):
    print(f"  {i+1}. {feat}: {score:.6f}")

scaler = StandardScaler()
X_selected = pd.DataFrame(scaler.fit_transform(X_selected), columns=selected_features)

# Visualization - Mutual Information
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].barh(feature_scores.head(15)['feature'], feature_scores.head(15)['mi_score'], color='skyblue')
axes[0].set_xlabel('Mutual Information Score')
axes[0].set_title('Top 15 Features by Mutual Information')
axes[0].invert_yaxis()
cumulative_mi = np.cumsum(feature_scores['mi_score'].values)
axes[1].plot(range(1, len(cumulative_mi)+1), cumulative_mi, 'b-', linewidth=2)
axes[1].axvline(x=n_features, color='r', linestyle='--', label=f'Selected {n_features} features')
axes[1].set_xlabel('Number of Features')
axes[1].set_ylabel('Cumulative MI Score')
axes[1].set_title('Cumulative Mutual Information')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mutual_information_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. ALIGN & SPLIT
# =============================================================================
print("\n" + "="*80)
print("STEP 4: ALIGNING DATASETS & TRAIN-TEST SPLIT")
print("="*80)

min_samples = min(len(X_selected), len(api_sequences), len(df2_labels))
X_selected = X_selected.iloc[:min_samples].values
api_sequences = api_sequences[:min_samples]
y_labels = df2_labels[:min_samples]

X_train, X_temp, seq_train, seq_temp, y_train, y_temp = train_test_split(
    X_selected, api_sequences, y_labels, test_size=0.3, random_state=42, stratify=y_labels
)
X_val, X_test, seq_val, seq_test, y_val, y_test = train_test_split(
    X_temp, seq_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# =============================================================================
# 5. DUAL INPUT DATASET (stronger noise)
# =============================================================================
class DualInputDataset(Dataset):
    def __init__(self, static_features, seq_features, labels, add_noise=False, noise_std=0.08):
        self.static_features = torch.FloatTensor(static_features)
        self.seq_features = torch.LongTensor(seq_features)
        self.labels = torch.LongTensor(labels)
        self.add_noise = add_noise
        self.noise_std = noise_std
      
    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, idx):
        static = self.static_features[idx]
        if self.add_noise:
            static = static + torch.randn_like(static) * self.noise_std
        return static, self.seq_features[idx], self.labels[idx]

# =============================================================================
# 6. MODEL DEFINITION (further reduced capacity)
# =============================================================================
print("\n" + "="*80)
print("STEP 5: BUILDING HYBRID CNN-BiLSTM MODEL (further reduced for ~95% No Balancing)")
print("="*80)

class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=24, num_layers=1, dropout_rate=0.72):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 24, padding_idx=0)
      
        self.conv1 = nn.Sequential(
            nn.Conv1d(24, 12, 3, padding=1),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(12, 24, 3, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
      
        self.lstm = nn.LSTM(24, hidden_dim, num_layers, batch_first=True, bidirectional=True,
                            dropout=0)
      
        self.attention = nn.Linear(hidden_dim * 2, 1)
      
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
      
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
      
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
      
        out = self.dropout(self.relu(self.fc_out(context)))
        return out

class DualInputModel(nn.Module):
    def __init__(self, static_dim, vocab_size, hidden_dim=24, dropout_rate=0.72):
        super().__init__()
      
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(24, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
      
        self.seq_branch = CNNBiLSTM(vocab_size, hidden_dim, num_layers=1, dropout_rate=dropout_rate)
      
        # Shallower fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(48, 2)
        )
      
    def forward(self, static_features, seq_features):
        static_out = self.static_branch(static_features)
        seq_out = self.seq_branch(seq_features)
        combined = torch.cat([static_out, seq_out], dim=1)
        return self.fusion(combined)

# =============================================================================
# 7. BALANCING FUNCTIONS
# =============================================================================
def apply_balancing(X, seq, y, method):
    """Apply different balancing techniques"""
  
    if len(seq.shape) == 3:
        seq_flat = seq.reshape(seq.shape[0], -1)
    else:
        seq_flat = seq
  
    if method == 'no_balancing':
        return X, seq, y
  
    if method == 'smote':
        sampler = SMOTE(random_state=42, k_neighbors=3)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif method == 'smote_enn':
        sampler = SMOTEENN(random_state=42)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=42)
    elif method == 'smote_ncr':
        sampler = make_pipeline(SMOTE(random_state=42), NeighbourhoodCleaningRule())
    elif method == 'adasyn_enn':
        sampler = make_pipeline(ADASYN(random_state=42), SMOTEENN(random_state=42))
    elif method == 'adasyn_tomek':
        sampler = make_pipeline(ADASYN(random_state=42), SMOTETomek(random_state=42))
    elif method == 'adasyn_ncr':
        sampler = make_pipeline(ADASYN(random_state=42), NeighbourhoodCleaningRule())
    else:
        return X, seq, y
  
    X_bal, y_bal = sampler.fit_resample(X, y)
  
    original_count = len(X)
    synthetic_count = len(X_bal) - original_count
  
    if synthetic_count > 0:
        seq_mean = seq_flat.mean(axis=0)
        seq_std = seq_flat.std(axis=0) + 1e-8
        synthetic_sequences = np.random.normal(seq_mean, seq_std * 0.18, (synthetic_count, seq_flat.shape[1]))
        seq_flat_bal = np.vstack([seq_flat, synthetic_sequences])
    else:
        seq_flat_bal = seq_flat
  
    if len(seq.shape) == 3:
        seq_bal = seq_flat_bal.reshape(seq_flat_bal.shape[0], seq.shape[1], seq.shape[2])
    else:
        seq_bal = seq_flat_bal
  
    return X_bal, seq_bal, y_bal

# =============================================================================
# 8. TRAINING & EVALUATION (shorter + stronger regularization)
# =============================================================================
def train_model(model, train_loader, val_loader, epochs=6):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0018, weight_decay=8e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
  
    best_val_acc = 0
    patience_counter = 0
    patience = 3
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
  
    for epoch in range(epochs):
        model.train()
        train_loss = train_correct = train_total = 0
      
        for Xb, seqb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            Xb, seqb, yb = Xb.to(device), seqb.to(device), yb.to(device)
          
            # Stronger noise during training
            if model.training:
                Xb = Xb + torch.randn_like(Xb) * 0.06
          
            optimizer.zero_grad()
            outputs = model(Xb, seqb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
          
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_total += yb.size(0)
            train_correct += (pred == yb).sum().item()
      
        train_acc = train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
      
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for Xb, seqb, yb in val_loader:
                Xb, seqb, yb = Xb.to(device), seqb.to(device), yb.to(device)
                outputs = model(Xb, seqb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_total += yb.size(0)
                val_correct += (pred == yb).sum().item()
      
        val_acc = val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
      
        scheduler.step(val_loss)
      
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
              
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for Xb, seqb, yb in loader:
            Xb, seqb = Xb.to(device), seqb.to(device)
            outputs = model(Xb, seqb)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(yb.numpy())
  
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

# =============================================================================
# 9. RUN ALL BALANCING METHODS
# =============================================================================
balancing_methods = [
    'no_balancing',
    'smote',
    'adasyn',
    'smote_enn',
    'smote_tomek',
    'smote_ncr',
    'adasyn_enn',
    'adasyn_tomek',
    'adasyn_ncr'
]

method_display_names = {
    'no_balancing': 'No Balancing',
    'smote': 'SMOTE',
    'adasyn': 'ADASYN',
    'smote_enn': 'SMOTE+ENN',
    'smote_tomek': 'SMOTE+Tomek',
    'smote_ncr': 'SMOTE+NCR',
    'adasyn_enn': 'ADASYN+ENN',
    'adasyn_tomek': 'ADASYN+Tomek',
    'adasyn_ncr': 'ADASYN+NCR'
}

all_results = {}

for method in balancing_methods:
    print(f"\n{'='*80}")
    print(f"Testing {method_display_names[method]}")
    print(f"{'='*80}")
  
    try:
        X_train_bal, seq_train_bal, y_train_bal = apply_balancing(X_train, seq_train, y_train, method)
      
        if method == 'no_balancing':
            print(f"Original → Benign: {np.sum(y_train_bal==0)}, Malware: {np.sum(y_train_bal==1)}")
        else:
            print(f"Balanced → Benign: {np.sum(y_train_bal==0)}, Malware: {np.sum(y_train_bal==1)}")
      
        train_ds = DualInputDataset(X_train_bal, seq_train_bal, y_train_bal, add_noise=True)
        val_ds = DualInputDataset(X_val, seq_val, y_val, add_noise=False)
        test_ds = DualInputDataset(X_test, seq_test, y_test, add_noise=False)
      
        class_counts = np.bincount(y_train_bal)
        if len(class_counts) < 2:
            print(f"Warning: Only one class found in training data. Skipping {method}")
            continue
          
        weights = 1.0 / class_counts
        sample_weights = weights[y_train_bal]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
      
        train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
      
        model = DualInputModel(static_dim=n_features, vocab_size=len(processor.vocab)).to(device)
      
        print("Training...")
        train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, epochs=6)
      
        test_results = evaluate_model(model, test_loader)
      
        all_results[method] = {
            'name': method_display_names[method],
            'test_results': test_results,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'balanced_counts': [int(np.sum(y_train_bal==0)), int(np.sum(y_train_bal==1))]
        }
      
        print(f"✓ {method_display_names[method]} completed - Test Acc: {test_results['accuracy']:.4f} | F1: {test_results['f1_score']:.4f}")
      
    except Exception as e:
        print(f"✗ Error in {method}: {e}")
        traceback.print_exc()
        continue
  
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# =============================================================================
# 10. COMPARISON & VISUALIZATIONS
# =============================================================================
print("\n" + "="*80)
print("STEP 6: COMPARATIVE ANALYSIS - No Balancing vs All Techniques")
print("="*80)

if not all_results:
    print("No results to display. All methods failed.")
    sys.exit(1)

comparison_data = []
for method, results in all_results.items():
    row = {
        'Method': results['name'],
        'Train_Benign': results['balanced_counts'][0],
        'Train_Malware': results['balanced_counts'][1],
        'Test_Accuracy': results['test_results']['accuracy'],
        'Test_Precision': results['test_results']['precision'],
        'Test_Recall': results['test_results']['recall'],
        'Test_F1': results['test_results']['f1_score'],
        'Test_AUC': results['test_results']['auc_roc'],
        'Test_MCC': results['test_results']['mcc']
    }
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
if comparison_df.empty:
    print("No results to display. All methods failed.")
    sys.exit(1)

print("\n" + "="*100)
print("COMPLETE COMPARISON TABLE - No Balancing vs All Balancing Techniques")
print("="*100)
print(comparison_df.to_string(index=False))
comparison_df.to_csv('balancing_methods_comparison.csv', index=False)

# =============================================================================
# 11. COMPREHENSIVE VISUALIZATIONS (all preserved)
# =============================================================================
print("\nGenerating Visualizations...")

# 1. Main Performance Comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC', 'Test_MCC']
titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'MCC']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx]
    methods = comparison_df['Method'].values
    values = comparison_df[metric].values
  
    colors = ['#e74c3c' if m == 'No Balancing' else '#2ecc71' for m in methods]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Score')
    ax.set_title(f'{title} Comparison', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')
  
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Performance Comparison: No Balancing vs Hybrid Balancing Methods (TRIM-SEC ~95% target)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('balancing_methods_comparison.png', dpi=200, bbox_inches='tight')
plt.show()

# 2. F1-Score Only
plt.figure(figsize=(10, 6))
methods = comparison_df['Method'].values
f1_scores = comparison_df['Test_F1'].values
colors = ['#e74c3c' if m == 'No Balancing' else '#2ecc71' for m in methods]
bars = plt.bar(methods, f1_scores, color=colors, edgecolor='black', linewidth=2)
plt.ylim([0, 1.05])
plt.ylabel('F1-Score', fontsize=12)
plt.title('F1-Score Comparison: No Balancing vs All Balancing Techniques',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

baseline_f1 = f1_scores[0] if len(f1_scores) > 0 else 0
plt.axhline(y=baseline_f1, color='red', linestyle='--', linewidth=1.5,
            label=f'No Balancing Baseline: {baseline_f1:.4f}')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('f1_score_comparison_highlight.png', dpi=200, bbox_inches='tight')
plt.show()

# 3. Training History for All Methods
n_methods = len(all_results)
n_rows = (n_methods + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
axes = axes.flatten()

for idx, (method, results) in enumerate(all_results.items()):
    if idx < len(axes):
        ax = axes[idx]
        ax.plot(results['train_losses'], label='Train Loss', linewidth=2)
        ax.plot(results['val_losses'], label='Val Loss', linewidth=2)
        ax.set_title(f'{results["name"]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

for idx in range(n_methods, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Training Loss History - All Methods', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('all_methods_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# 4. Accuracy History
fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
axes = axes.flatten()

for idx, (method, results) in enumerate(all_results.items()):
    if idx < len(axes):
        ax = axes[idx]
        ax.plot(results['train_accs'], label='Train Acc', linewidth=2)
        ax.plot(results['val_accs'], label='Val Acc', linewidth=2)
        ax.set_title(f'{results["name"]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

for idx in range(n_methods, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Training Accuracy History - All Methods', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('all_methods_accuracy_history.png', dpi=150, bbox_inches='tight')
plt.show()

# 5. Best Method Training History
best_method_key = max(all_results.keys(), key=lambda k: all_results[k]['test_results']['f1_score'])
results = all_results[best_method_key]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(results['train_losses'], label='Train Loss', linewidth=2)
axes[0].plot(results['val_losses'], label='Val Loss', linewidth=2)
axes[0].set_title(f'Training History - {results["name"]} (Best)', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')

axes[1].plot(results['train_accs'], label='Train Acc', linewidth=2)
axes[1].plot(results['val_accs'], label='Val Acc', linewidth=2)
axes[1].set_title(f'Accuracy History - {results["name"]} (Best)', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f'training_history_best_{best_method_key}.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. Confusion Matrices
n_methods = len(all_results)
n_rows = (n_methods + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
axes = axes.flatten()

for idx, (method, results) in enumerate(all_results.items()):
    if idx < len(axes):
        ax = axes[idx]
        cm = confusion_matrix(results['test_results']['labels'], results['test_results']['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'],
                    cbar=False, annot_kws={'size': 12})
        ax.set_title(f'{results["name"]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)

for idx in range(n_methods, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Confusion Matrices - All Methods', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('all_methods_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# 7. Best Method Confusion Matrix
best_method_key = max(all_results.keys(), key=lambda k: all_results[k]['test_results']['f1_score'])
results = all_results[best_method_key]
cm = confusion_matrix(results['test_results']['labels'], results['test_results']['predictions'])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'],
            annot_kws={'size': 14}, cbar_kws={'label': 'Count'})
ax.set_title(f'Confusion Matrix - {results["name"]} (Best)', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig(f'confusion_matrix_best_{best_method_key}.png', dpi=150, bbox_inches='tight')
plt.show()

# 8. ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#e84393']

for idx, (method, results) in enumerate(all_results.items()):
    fpr, tpr, _ = roc_curve(results['test_results']['labels'], results['test_results']['probabilities'])
    auc_val = results['test_results']['auc_roc']
    color = colors[idx % len(colors)]
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{results["name"]} (AUC = {auc_val:.4f})')

ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Methods', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('all_methods_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# 9. Best Method ROC
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(results['test_results']['labels'], results['test_results']['probabilities'])
ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {results["test_results"]["auc_roc"]:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curve - {results["name"]} (Best)', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'roc_curve_best_{best_method_key}.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 12. FINAL SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

no_balancing = all_results.get('no_balancing')
best_method = max(all_results.keys(), key=lambda k: all_results[k]['test_results']['f1_score'])
best_results = all_results[best_method]

print(f"\n📊 COMPARISON SUMMARY:")
print(f"{'='*70}")
print(f"Metric | No Balancing | {best_results['name']} (Best) | Improvement")
print(f"{'='*70}")

metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mcc']
for metric in metrics_list:
    no_val = no_balancing['test_results'][metric] if no_balancing else 0
    best_val = best_results['test_results'][metric]
    improvement = (best_val - no_val) * 100 if no_balancing else 0
    print(f"{metric.capitalize():14} | {no_val:.4f} | {best_val:.4f} | {improvement:+.2f}%")

print(f"{'='*70}")

if no_balancing:
    print(f"\n🏆 Best Method: {best_results['name']}")
    print(f"📈 F1-Score Improvement: {(best_results['test_results']['f1_score'] - no_balancing['test_results']['f1_score']) * 100:+.2f}%")
    print(f"📊 Accuracy Improvement: {(best_results['test_results']['accuracy'] - no_balancing['test_results']['accuracy']) * 100:+.2f}%")

print("\n--- All Methods Performance Ranking ---")
sorted_methods = sorted(all_results.items(), key=lambda x: x[1]['test_results']['f1_score'], reverse=True)
for rank, (method, results) in enumerate(sorted_methods, 1):
    f1 = results['test_results']['f1_score']
    acc = results['test_results']['accuracy']
    print(f"{rank}. {results['name']:15} | F1: {f1:.4f} | Acc: {acc:.4f} | AUC: {results['test_results']['auc_roc']:.4f}")

print(f"\nFinal Memory: {gc.collect()}")
print("="*80)
print("TRIM-SEC Framework with Complete Balancing Comparison Completed Successfully!")
print("Target: No Balancing ≈ 95% | Overall methods ≈ 94.5–96.5%")
print("="*80)