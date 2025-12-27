#!/usr/bin/env python3
"""
Training script for ResLiteAudioCNN with Cross-Validation
"""
import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import time
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import dct

import warnings
warnings.filterwarnings('ignore')

# Import config and model
from config import *
from model import ResLiteAudioCNN

# Import data utilities
from utils import (
    cache_dataset_to_npy,
    CachedMelDataset,
    UrbanSound8KDataset,
    mixup_data,
    cutmix_data,
    apply_smote_oversampling
)





# ============================================================
# Training Functions (Data utilities imported from utils.py)
# ============================================================

def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, 
                use_mixup=USE_MIXUP, use_cutmix=USE_CUTMIX, use_salience_weighting=False):
    """Train for one epoch with optional salience-based sample weighting"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch (with or without salience)
        if use_salience_weighting and len(batch_data) == 3:
            data, target, salience = batch_data
            data, target, salience = data.to(device), target.to(device), salience.to(device)
            
            # Compute sample weights based on salience
            # salience=1 (foreground) -> weight=1.0
            # salience=2 (background) -> weight=BACKGROUND_WEIGHT
            sample_weights = torch.where(salience == 1, 1.0, BACKGROUND_WEIGHT)
        else:
            data, target = batch_data[0], batch_data[1]
            data, target = data.to(device), target.to(device)
            sample_weights = None
        
        # Apply Mixup or CutMix
        if use_mixup and np.random.rand() < MIXUP_PROB:
            data, target_a, target_b, lam = mixup_data(data, target)
            
            with autocast():
                output = model(data)
                loss_a = F.cross_entropy(output, target_a, reduction='none')
                loss_b = F.cross_entropy(output, target_b, reduction='none')
                
                # Apply sample weights if available
                if sample_weights is not None:
                    loss = (lam * loss_a * sample_weights + (1 - lam) * loss_b * sample_weights).mean()
                else:
                    loss = (lam * loss_a + (1 - lam) * loss_b).mean()
        elif use_cutmix and np.random.rand() < CUTMIX_PROB:
            data, target_a, target_b, lam = cutmix_data(data, target)
            
            with autocast():
                output = model(data)
                loss_a = F.cross_entropy(output, target_a, reduction='none')
                loss_b = F.cross_entropy(output, target_b, reduction='none')
                
                # Apply sample weights if available
                if sample_weights is not None:
                    loss = (lam * loss_a * sample_weights + (1 - lam) * loss_b * sample_weights).mean()
                else:
                    loss = (lam * loss_a + (1 - lam) * loss_b).mean()
        else:
            with autocast():
                output = model(data)
                if sample_weights is not None:
                    # Per-sample weighted loss
                    loss_per_sample = F.cross_entropy(output, target, reduction='none')
                    loss = (loss_per_sample * sample_weights).mean()
                else:
                    loss = criterion(output, target)
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update scheduler (OneCycleLR steps per batch)
        scheduler.step()
        
        # Metrics
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    epoch_score = 0.8 * epoch_acc + 0.2 * epoch_f1  # Weighted score
    
    return epoch_loss, epoch_acc, epoch_f1, epoch_score


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    val_score = 0.8 * val_acc + 0.2 * val_f1  # Weighted score
    
    return val_loss, val_acc, val_f1, val_score, all_preds, all_targets


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Macro F1
    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Val F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Macro F1')
    axes[1, 0].set_title('Training and Validation Macro F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Weighted Score (0.8 * Acc + 0.2 * F1)
    axes[1, 1].plot(history['train_score'], label='Train Score')
    axes[1, 1].plot(history['val_score'], label='Val Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Weighted Score')
    axes[1, 1].set_title('Weighted Score (0.8*Acc + 0.2*F1)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(targets, preds, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(targets, preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Main Training Function
# ============================================================

def train_fold(fold_idx, train_df, val_df, cache_dir, output_dir, device, 
               use_cached=True, num_epochs=NUM_EPOCHS):
    """
    Train a single fold
    
    Args:
        fold_idx: Fold index (for logging)
        train_df: Training DataFrame
        val_df: Validation DataFrame
        cache_dir: Directory with cached .npy files
        output_dir: Directory to save outputs
        device: torch device
        use_cached: Whether to use cached data
        num_epochs: Number of epochs
    """
    print("\n" + "=" * 80)
    # if use_fusion:
    #     print(f"Training Fold {fold_idx} - FusedCNN (GFCC+STE)")
    #     if norm_stats is not None:
    #         print("  Using feature-wise normalization")
    # else:
    print(f"Training Fold {fold_idx} - ResLiteAudioCNN")
    print("=" * 80)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Check and display salience weighting status
    if USE_SALIENCE_WEIGHTING:
        if 'salience' in train_df.columns:
            fg_count = (train_df['salience'] == 1).sum()
            bg_count = (train_df['salience'] == 2).sum()
            print(f"\n Salience-based sample weighting ENABLED")
            print(f"  - Foreground samples (salience=1): {fg_count} ({fg_count/len(train_df)*100:.1f}%) - weight=1.0")
            print(f"  - Background samples (salience=2): {bg_count} ({bg_count/len(train_df)*100:.1f}%) - weight={BACKGROUND_WEIGHT}")
        else:
            print(f"\n WARNING: USE_SALIENCE_WEIGHTING=True but 'salience' column not found in dataset!")
            print(f"  Salience weighting will be disabled for this run.")
    else:
        print(f"\nSalience-based sample weighting: Disabled")
    
    # Create datasets
    train_dataset = CachedMelDataset(train_df, mode='train', augment=True, 
                                     return_salience=USE_SALIENCE_WEIGHTING)
    val_dataset = CachedMelDataset(val_df, mode='val', augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model - Choose based on feature type (fusion disabled)
    # if use_fusion:
    #     from model import FusedCNN
    #     model = FusedCNN(num_classes=10, dropout=0.3).to(device)
    #     model_name = "fused"
    # else:
    model = ResLiteAudioCNN(num_classes=10, dropout=0.3).to(device)
    model_name = "reslite"
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=WEIGHT_DECAY)
    
    # OneCycleLR with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,  # initial_lr = max_lr / div_factor
        final_div_factor=1e4
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_score': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_score': []
    }
    
    best_val_score = 0.0
    best_epoch = 0
    patience = 0
    max_patience = 10
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Train
        train_loss, train_acc, train_f1, train_score = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            use_mixup=USE_MIXUP, use_cutmix=USE_CUTMIX, use_salience_weighting=USE_SALIENCE_WEIGHTING
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_score, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_score'].append(train_score)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_score'].append(val_score)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Score: {train_score:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Score: {val_score:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model based on weighted score
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 1
            patience = 0
            
            # Save checkpoint
            checkpoint_path = output_dir / f"{model_name}_valfolds{fold_idx}_best.pth"
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_score': val_score,
                'model_type': 'ResLiteAudioCNN'  # Fusion disabled
            }, checkpoint_path)
            
            print(f"✓ Saved best model (Val Score: {val_score:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Plot results
    plot_training_history(history, output_dir / f"{model_name}_valfolds{fold_idx}_history.png")
    
    # Load best model and evaluate
    checkpoint = torch.load(output_dir / f"{model_name}_valfolds{fold_idx}_best.pth")
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_acc, val_f1, val_score, val_preds, val_targets = validate(
        model, val_loader, criterion, device
    )
    
    # Classification report
    print("\nClassification Report:")
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    print(classification_report(val_targets, val_preds, target_names=class_names, digits=4))
    
    # Confusion matrix
    plot_confusion_matrix(val_targets, val_preds, class_names, 
                         output_dir / f"{model_name}_valfolds{fold_idx}_confusion_matrix.png")
    
    print(f"\n✓ Fold {fold_idx} completed - Best Val Score: {best_val_score:.4f} (Acc: {val_acc:.4f}, F1: {val_f1:.4f}) at epoch {best_epoch}")
    
    return best_val_score, best_epoch


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train ResLiteAudioCNN or FusedCNN with Cross-Validation')
    parser.add_argument('--csv', type=str, default='Kaggle_Data/metadata/kaggle_train.csv',
                       help='Path to training CSV')
    parser.add_argument('--audio-dir', type=str, default='Kaggle_Data/audio',
                       help='Path to audio directory')
    parser.add_argument('--cache-dir', type=str, default='cache/mel_spectrograms_augmented',
                       help='Directory to cache mel spectrograms')
    parser.add_argument('--output-dir', type=str, default='outputs_reslite_final',
                       help='Directory to save outputs')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU IDs to use (e.g., "0" or "0,1,2,3")')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--no-cache', action='store_true',
                       help='Skip caching and load audio on-the-fly')
    parser.add_argument('--val-folds', type=str, default='8',
                       help='Validation fold(s), comma-separated (e.g., "8" or "7,8" or "1,2,3,4,5,6,7,8" for full 8-fold CV). Only used when --split-mode=fold')
    parser.add_argument('--split-mode', type=str, default='fold', choices=['fold', 'random'],
                       help='Train/val split mode: "fold" for fold-based split (train 7 folds, val 1 fold), "random" for random reshuffle with custom ratio')
    parser.add_argument('--val-ratio', type=float, default=0.125,
                       help='Validation ratio (0.0-1.0) when using --split-mode=random. Default: 0.125 (1/8, same as 1 fold)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for random split mode. Default: 42')
    parser.add_argument('--use-augmented', action='store_true',
                       help='Use 3x augmented audio dataset (original + stretch + pitch_shift). Automatically sets --csv and --audio-dir to augmented data paths.')
    # parser.add_argument('--use-fusion', action='store_true',
    #                    help='Use GFCC+STE fusion features instead of mel spectrograms')  # COMMENTED OUT
    

    # === 新增代码 ===
    parser.add_argument('--data-fraction', type=float, default=1.0,
                       help='Fraction of dataset to use (e.g., 0.7 for 70%). The rest is discarded.')
    parser.add_argument('--use-smote', action='store_true',
                       help='Apply SMOTE oversampling to minority classes (car_horn and gun_shot)')
    parser.add_argument('--smote-ratio', type=float, default=1.0,
                       help='SMOTE target ratio (1.0=full balance, 0.5=50%% of majority). Default: 1.0')
    # =================
    
    args = parser.parse_args()
    
    # Auto-set paths for augmented data if flag is set
    if args.use_augmented:
        if args.csv == 'Kaggle_Data/metadata/kaggle_train.csv':  # Only override if using default
            args.csv = 'Data_Augmented/metadata/train_augmented.csv'
        if args.audio_dir == 'Kaggle_Data/audio':  # Only override if using default
            args.audio_dir = 'Data_Augmented/audio'
        if 'augmented' not in args.cache_dir:  # Update cache dir to avoid mixing with non-augmented cache
            args.cache_dir = args.cache_dir.replace('mel_spectrograms', 'mel_spectrograms_augmented')
        print(f"\n{'='*80}")
        print("Using 3x Augmented Audio Dataset")
        print(f"  CSV: {args.csv}")
        print(f"  Audio Dir: {args.audio_dir}")
        print(f"  Cache Dir: {args.cache_dir}")
        print(f"{'='*80}\n")
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(args.csv)
    print(f"\nTotal samples: {len(df)}")
    # print(f"Folds: {sorted(df['fold'].unique())}")
    
    # Display feature type (fusion disabled)
    # if args.use_fusion:
    #     print(f"\nFeature type: GFCC+STE Fusion")
    #     print(f"  GFCC parameters:")
    #     print(f"    - n_filters: {GFCC_N_FILTERS}")
    #     print(f"    - n_ceps: {GFCC_N_CEPS}")
    #     print(f"    - win_time: {GFCC_WIN_TIME}s")
    #     print(f"    - hop_time: {GFCC_HOP_TIME}s")
    #     # Update cache dir for fusion features
    #     if 'fusion' not in args.cache_dir:
    #         args.cache_dir = args.cache_dir.replace('mel_spectrograms', 'fusion_features')
    # else:
    print(f"\nFeature type: {'PCEN' if USE_PCEN else 'Log-Mel'} Spectrogram")
    if USE_PCEN:
        print(f"  PCEN parameters:")
        print(f"    - gain: {PCEN_GAIN}")
        print(f"    - bias: {PCEN_BIAS}")
        print(f"    - power: {PCEN_POWER}")
        print(f"    - time_constant: {PCEN_TIME_CONSTANT}")
    
    # Display salience weighting configuration
    print(f"\nSalience-based Sample Weighting: {'ENABLED' if USE_SALIENCE_WEIGHTING else 'DISABLED'}")
    if USE_SALIENCE_WEIGHTING:
        print(f"  - Background weight (salience=2): {BACKGROUND_WEIGHT}")
        print(f"  - Note: Lower weight = less influence from noisy/background samples")
    
    # Display SMOTE configuration
    print(f"\nSMOTE Oversampling: {'ENABLED' if args.use_smote else 'DISABLED'}")
    if args.use_smote:
        print(f"  - Target classes: car_horn (classID=1), gun_shot (classID=6)")
        print(f"  - Target ratio: {args.smote_ratio} (1.0=full balance, 0.5=50% of majority)")
        print(f"  - Random seed: {args.random_seed}")
    
    # Cache dataset if not skipping
    if not args.no_cache:
        df_cached = cache_dataset_to_npy(
            args.csv, 
            args.audio_dir, 
            args.cache_dir, 
            use_pcen=USE_PCEN
        )
    else:
        print("\nSkipping caching, will load audio on-the-fly")
        df_cached = df
    
    # === 新增代码：数据采样 ===
    if args.data_fraction < 1.0:
        print(f"\n" + "=" * 80)
        print(f"Subsampling Dataset: Keeping {args.data_fraction*100:.1f}% of data")
        
        # 确定分层列 (优先使用 classID, 如果没有则不分层)
        stratify_col = None
        if 'classID' in df_cached.columns:
            stratify_col = df_cached['classID']
        elif 'Class' in df_cached.columns: 
            stratify_col = df_cached['Class']
            
        # 使用 train_test_split 进行随机采样，保留 train 部分作为新的数据集
        df_cached, _ = train_test_split(
            df_cached,
            train_size=args.data_fraction,
            random_state=args.random_seed,
            stratify=stratify_col,
            shuffle=True
        )
        df_cached = df_cached.reset_index(drop=True)
        print(f"New dataset size: {len(df_cached)} samples")
        print("=" * 80 + "\n")
    # ===========================
    
    # === Apply SMOTE oversampling if requested ===
    if args.use_smote:
        df_cached = apply_smote_oversampling(
            df_cached, 
            cache_dir=args.cache_dir,
            minority_classes=[1, 6],  # car_horn=1, gun_shot=6
            target_ratio=args.smote_ratio,
            random_state=args.random_seed
        )
    # =============================================

    print(f"df_cached: {df_cached.head()}")
    # Split data based on split mode
    if args.split_mode == 'fold':
        # Mode 1: Fold-based split (train 7 folds, val 1 fold)
        val_folds = [int(f.strip()) for f in args.val_folds.split(',')]
        val_folds_str = '_'.join(map(str, val_folds))
        
        print("\n" + "=" * 80)
        print("Training Strategy: Fold-based Split")
        print(f"  Split mode: {args.split_mode}")
        print(f"  Validation folds: {val_folds}")
        print(f"  Training folds: {[f for f in range(1, 9) if f not in val_folds]}")
        print(f"  Model: ResLiteAudioCNN")  # Fusion disabled
        print("=" * 80)
        
        # Split data by folds (no random shuffling, use original fold assignment)
        val_df = df_cached[df_cached['fold'].isin(val_folds)].reset_index(drop=True)
        train_df = df_cached[~df_cached['fold'].isin(val_folds)].reset_index(drop=True)
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_df)} samples (from folds {sorted(train_df['fold'].unique())})")
        print(f"  Val: {len(val_df)} samples (from folds {sorted(val_df['fold'].unique())})")
        
        fold_identifier = val_folds_str
    else:
        # Mode 2: Random reshuffle with custom train/val ratio
        print("\n" + "=" * 80)
        print("Training Strategy: Random Split with Custom Ratio")
        print(f"  Split mode: {args.split_mode}")
        print(f"  Validation ratio: {args.val_ratio:.3f} ({args.val_ratio*100:.1f}%)")
        print(f"  Random seed: {args.random_seed}")
        print(f"  Model: ResLiteAudioCNN")  # Fusion disabled
        print("=" * 80)
        
        # Use stratified split to maintain class distribution
        train_df, val_df = train_test_split(
            df_cached,
            test_size=args.val_ratio,
            random_state=args.random_seed,
            stratify=df_cached['classID'],  # Maintain class balance
            shuffle=True
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df_cached)*100:.1f}%)")
        print(f"  Val: {len(val_df)} samples ({len(val_df)/len(df_cached)*100:.1f}%)")
        # print(f"  Train folds: {sorted(train_df['fold'].unique())}")
        # print(f"  Val folds: {sorted(val_df['fold'].unique())}")
        
        # Create identifier for random split
        fold_identifier = f"random_{args.val_ratio:.3f}_seed{args.random_seed}"
    
    print(f"\nClass distribution in validation set:")
    print(val_df['classID'].value_counts().sort_index())
    
    print("\n" + "=" * 80)
    if args.split_mode == 'fold':
        print(f"Starting Training with Validation Folds: {val_folds}")
    else:
        print(f"Starting Training with Random Split (val_ratio={args.val_ratio:.3f})")
    print("=" * 80)
    
    # Train model
    best_score, best_epoch = train_fold(
        fold_idx=fold_identifier,  # Use fold_identifier as identifier
        train_df=train_df,
        val_df=val_df,
        cache_dir=args.cache_dir,
        output_dir=output_dir,
        device=device,
        use_cached=not args.no_cache,
        num_epochs=args.epochs
    )
    
    # Load checkpoint to get final metrics (fusion disabled)
    model_name = "reslite"  # Fusion disabled
    checkpoint = torch.load(output_dir / f"{model_name}_valfolds{fold_identifier}_best.pth")
    
    print("\n" + "=" * 80)
    print(f"Training Complete - ResLiteAudioCNN")  # Fusion disabled
    print("=" * 80)
    if args.split_mode == 'fold':
        val_folds = [int(f.strip()) for f in args.val_folds.split(',')]
        print(f"Split mode: Fold-based")
        print(f"Validation folds: {val_folds}")
    else:
        print(f"Split mode: Random")
        print(f"Validation ratio: {args.val_ratio:.3f}")
        print(f"Random seed: {args.random_seed}")
    print(f"Best Validation Score: {best_score:.4f} (epoch {best_epoch})")
    print(f"  - Accuracy: {checkpoint['val_acc']:.4f}")
    print(f"  - Macro-F1: {checkpoint['val_f1']:.4f}")
    print(f"  - Weighted Score: {checkpoint['val_score']:.4f}")
    print(f"Model saved to: {output_dir}/{model_name}_valfolds{fold_identifier}_best.pth")
    print("=" * 80)


if __name__ == '__main__':
    main()







