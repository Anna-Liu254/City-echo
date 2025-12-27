#!/usr/bin/env python3
"""
Utilities for UrbanSound8K Dataset
Includes data loading, preprocessing, augmentation, and dataset classes
"""
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import os
from imblearn.over_sampling import SMOTE

from config import *




# ============================================================
# Helper Functions
# ============================================================

def log(mel_spec, eps=1e-10):
    """Convert mel spectrogram to log scale"""
    return np.log10(mel_spec + eps)


# ============================================================
# Audio Loading and Feature Extraction
# ============================================================

def load_audio_with_circular_padding(audio_path, target_length, sr=SAMPLE_RATE):
    """
    Load audio and apply circular padding if too short
    
    Args:
        audio_path: Path to audio file
        target_length: Target length in samples
        sr: Sample rate
    
    Returns:
        audio: Padded audio array
    """
    # Load audio
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    
    current_length = len(y)
    
    if current_length < target_length:
        # Circular padding (repeat the audio)
        num_repeats = (target_length // current_length) + 1
        y = np.tile(y, num_repeats)
        y = y[:target_length]
    elif current_length > target_length:
        # Random crop (center crop for consistency in caching)
        start = (current_length - target_length) // 2
        y = y[start:start + target_length]
    
    return y


def audio_to_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS, 
                             n_fft=N_FFT, hop_length=HOP_LENGTH, use_pcen=USE_PCEN):
    """
    Convert audio to mel spectrogram with optional PCEN
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        use_pcen: If True, use PCEN instead of Log-Mel
    
    Returns:
        mel_spec: Mel spectrogram with shape (1, n_mels, time)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=8000
    )
    
    # Apply PCEN or Log-Mel transformation
    if use_pcen:
        # ---------------------------------------------------------
        # PCEN (Per-Channel Energy Normalization)
        # Better for noisy environments like urban sounds
        # ---------------------------------------------------------
        mel_spec_processed = librosa.pcen(
            mel_spec * (2**31),  # Scale input to match PCEN expected range
            sr=sr,
            hop_length=hop_length,
            gain=PCEN_GAIN,          # AGC strength
            bias=PCEN_BIAS,          # Prevents division by zero
            power=PCEN_POWER,        # Compression exponent
            time_constant=PCEN_TIME_CONSTANT,  # AGC time constant
            eps=PCEN_EPS
        )
    else:
        # Log-Mel (traditional approach)
        mel_spec_processed = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Add channel dimension: (1, n_mels, time)
    mel_spec_processed = mel_spec_processed[np.newaxis, ...]
    
    return mel_spec_processed.astype(np.float32)


# ============================================================
# HPSS Feature Extraction - COMMENTED OUT (Use single channel instead)
# ============================================================
# def extract_hpss_mel_spectrogram(audio_path, target_length, sr=SAMPLE_RATE, 
#                                   n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
#     """
#     Extract 3-channel HPSS mel spectrogram: [full, harmonic, percussive]
#     
#     Args:
#         audio_path: Path to audio file
#         target_length: Target length in samples
#         sr: Sample rate
#         n_mels: Number of mel bins
#         n_fft: FFT window size
#         hop_length: Hop length for STFT
#     
#     Returns:
#         mel_spec: 3-channel mel spectrogram with shape (3, n_mels, time)
#     """
#     # Load and pad audio
#     y = load_audio_with_circular_padding(audio_path, target_length, sr)
#     
#     # Compute STFT and HPSS decomposition
#     D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
#     H, P = librosa.decompose.hpss(D)
#     
#     # Compute mel spectrograms for each component
#     mel_full = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, 
#                                                n_fft=n_fft, hop_length=hop_length, fmax=8000)
#     mel_h = librosa.feature.melspectrogram(S=np.abs(H)**2, sr=sr, n_mels=n_mels, 
#                                            n_fft=n_fft, hop_length=hop_length, fmax=8000)
#     mel_p = librosa.feature.melspectrogram(S=np.abs(P)**2, sr=sr, n_mels=n_mels, 
#                                            n_fft=n_fft, hop_length=hop_length, fmax=8000)
#     
#     # Apply log transformation and stack
#     X = np.stack([log(mel_full), log(mel_h), log(mel_p)], axis=0)
#     
#     return X


# ============================================================
# Data Caching
# ============================================================

def cache_dataset_to_npy(csv_path, audio_dir, cache_dir, folds=None, use_pcen=USE_PCEN):
    """
    Cache all audio files as mel spectrograms to .npy files
    
    Args:
        csv_path: Path to CSV file
        audio_dir: Path to audio directory
        cache_dir: Directory to save cached .npy files
        folds: List of folds to cache (None for all)
        use_pcen: If True, use PCEN instead of Log-Mel
    
    Returns:
        DataFrame with cached file paths
    """
    print("\n" + "=" * 80)
    print(f"Caching Dataset to NPY Files ({'PCEN' if use_pcen else 'Log-Mel'})")
    print("=" * 80)
    
    # Create cache directory
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter by folds if specified
    if folds is not None:
        df = df[df['fold'].isin(folds)]
    
    print(f"Total files to cache: {len(df)}")
    
    # Target length
    target_length = int(SAMPLE_RATE * DURATION)
    
    # Cache each file
    cached_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Caching"):
        fold = row['fold']
        filename = row['slice_file_name']
        
        # Audio path
        audio_path = Path(audio_dir) / f"fold{fold}" / filename
        
        # Cache path
        pcen_suffix = "_pcen" if use_pcen else ""
        cache_filename = f"train_{filename.replace('.wav', f'{pcen_suffix}.npy')}"
        cache_path = cache_dir / cache_filename
        
        # Check if already cached
        if cache_path.exists():
            cached_paths.append(str(cache_path))
            continue
        
        try:
            # Extract single-channel mel spectrogram
            audio = load_audio_with_circular_padding(audio_path, target_length)
            mel_spec = audio_to_mel_spectrogram(audio, use_pcen=use_pcen)
            # Save mel spectrogram: shape (1, n_mels, T)
            np.save(cache_path, mel_spec)
            
            cached_paths.append(str(cache_path))
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            cached_paths.append(None)
    
    # Add cached paths to dataframe
    df['cached_path'] = cached_paths
    df = df[df['cached_path'].notna()]  # Remove failed entries
    
    print(f"\n✓ Successfully cached {len(df)} files")
    
    return df


def apply_smote_oversampling(df, cache_dir, minority_classes=[1, 6], target_ratio=1.0, random_state=42):
    """
    Apply SMOTE oversampling to minority classes (car_horn and gun_shot)
    
    SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples
    by interpolating between existing minority class samples in feature space.
    
    Args:
        df: DataFrame with columns ['cached_path', 'classID', ...]
        cache_dir: Directory containing cached mel spectrograms
        minority_classes: List of class IDs to oversample (default: [1, 6] for car_horn and gun_shot)
        target_ratio: Target ratio for minority classes relative to majority class
                      1.0 = balance all classes equally
                      0.5 = make minority classes 50% of majority class size
        random_state: Random seed for reproducibility
    
    Returns:
        Updated DataFrame with synthetic samples added
    """
    
    print("\n" + "=" * 80)
    print("Applying SMOTE Oversampling to Minority Classes")
    print("=" * 80)
    
    cache_dir = Path(cache_dir)
    
    # Analyze class distribution
    class_counts = df['classID'].value_counts().sort_index()
    print("\nOriginal class distribution:")
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    for class_id, count in class_counts.items():
        marker = " 🎯" if class_id in minority_classes else ""
        print(f"  Class {class_id} ({class_names[class_id]:18s}): {count:4d} samples{marker}")
    
    # Determine majority class size for target calculation
    majority_count = class_counts.max()
    target_count = int(majority_count * target_ratio)
    
    print(f"\nSMOTE configuration:")
    print(f"  Minority classes to oversample: {minority_classes}")
    print(f"  Target ratio: {target_ratio}")
    print(f"  Majority class size: {majority_count}")
    print(f"  Target size for minority classes: {target_count}")
    
    # Process each minority class separately
    synthetic_records = []
    
    for class_id in minority_classes:
        class_df = df[df['classID'] == class_id]
        current_count = len(class_df)
        
        if current_count >= target_count:
            print(f"\nClass {class_id} ({class_names[class_id]}): {current_count} samples (already sufficient, skipping)")
            continue
        
        n_synthetic = target_count - current_count
        print(f"\nClass {class_id} ({class_names[class_id]}): Generating {n_synthetic} synthetic samples")
        
        # Load all mel spectrograms for this class
        mel_specs = []
        valid_indices = []
        
        for idx, row in class_df.iterrows():
            try:
                mel_spec = np.load(row['cached_path'])
                mel_specs.append(mel_spec)
                valid_indices.append(idx)
            except Exception as e:
                print(f"  Warning: Could not load {row['cached_path']}: {e}")
        
        if len(mel_specs) == 0:
            print(f"  Error: No valid samples found for class {class_id}")
            continue
        
        # Stack and get shape
        mel_specs = np.array(mel_specs)  # Shape: (n_samples, channels, n_mels, time)
        n_samples, channels, n_mels, time_steps = mel_specs.shape
        
        print(f"  Loaded {n_samples} samples with shape (channels={channels}, n_mels={n_mels}, time={time_steps})")
        
        # Flatten mel spectrograms for SMOTE
        # Shape: (n_samples, channels * n_mels * time_steps)
        X_flat = mel_specs.reshape(n_samples, -1)
        y = np.full(n_samples, class_id)
        
        # Apply SMOTE
        # We need to add dummy samples from other classes to satisfy SMOTE's requirement
        # of having at least 2 classes and k_neighbors < n_minority_samples
        k_neighbors = min(5, n_samples - 1) if n_samples > 1 else 1
        
        if n_samples < 2:
            print(f"  Warning: Only {n_samples} sample(s) for class {class_id}, cannot apply SMOTE (need at least 2)")
            continue
        
        try:
            # Create SMOTE sampler
            smote = SMOTE(
                sampling_strategy={class_id: target_count},
                k_neighbors=k_neighbors,
                random_state=random_state
            )
            
            # Add dummy majority class samples (we'll discard them later)
            dummy_count = max(2, n_samples)  # Ensure we have enough samples
            X_dummy = np.random.randn(dummy_count, X_flat.shape[1]) * 0.01
            y_dummy = np.full(dummy_count, 0)  # Use class 0 as dummy
            
            X_combined = np.vstack([X_flat, X_dummy])
            y_combined = np.concatenate([y, y_dummy])
            
            # Apply SMOTE
            X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)
            
            # Extract only the minority class samples (including synthetics)
            minority_mask = y_resampled == class_id
            X_minority = X_resampled[minority_mask]
            
            # Get only synthetic samples (excluding original samples)
            X_synthetic = X_minority[n_samples:]  # Skip original samples
            
            # Reshape back to mel spectrogram shape
            X_synthetic = X_synthetic.reshape(-1, channels, n_mels, time_steps)
            
            print(f"  Generated {len(X_synthetic)} synthetic samples")
            
            # Save synthetic samples to cache
            for i, synthetic_mel in enumerate(X_synthetic):
                # Create filename for synthetic sample
                synthetic_filename = f"synthetic_class{class_id}_sample{i}.npy"
                synthetic_path = cache_dir / synthetic_filename
                
                # Save to cache
                np.save(synthetic_path, synthetic_mel.astype(np.float32))
                
                # Create record for DataFrame
                # Copy other attributes from a random sample of this class
                template_row = class_df.sample(n=1, random_state=random_state + i).iloc[0]
                
                synthetic_record = {
                    'cached_path': str(synthetic_path),
                    'classID': class_id,
                    'slice_file_name': synthetic_filename,
                    'is_synthetic': True
                }
                
                # Copy other columns if they exist
                for col in df.columns:
                    if col not in synthetic_record and col in template_row:
                        synthetic_record[col] = template_row[col]
                
                synthetic_records.append(synthetic_record)
            
        except Exception as e:
            print(f"  Error applying SMOTE for class {class_id}: {e}")
            continue
    
    # Create DataFrame from synthetic records and concatenate with original
    if len(synthetic_records) > 0:
        synthetic_df = pd.DataFrame(synthetic_records)
        df_augmented = pd.concat([df, synthetic_df], ignore_index=True)
        
        print("\n" + "=" * 80)
        print("SMOTE Oversampling Complete")
        print("=" * 80)
        print(f"Original samples: {len(df)}")
        print(f"Synthetic samples: {len(synthetic_records)}")
        print(f"Total samples: {len(df_augmented)}")
        
        print("\nNew class distribution:")
        new_class_counts = df_augmented['classID'].value_counts().sort_index()
        for class_id, count in new_class_counts.items():
            original_count = class_counts.get(class_id, 0)
            added = count - original_count
            marker = f" (+{added} synthetic)" if added > 0 else ""
            print(f"  Class {class_id} ({class_names[class_id]:18s}): {count:4d} samples{marker}")
        
        return df_augmented
    else:
        print("\n⚠ No synthetic samples were generated")
        return df


# ============================================================
# Data Augmentation
# ============================================================

def add_gaussian_noise(mel_spec, noise_level=NOISE_LEVEL):
    """Add Gaussian noise to mel spectrogram"""
    noise = np.random.randn(*mel_spec.shape) * noise_level
    return mel_spec + noise


def freq_mask(mel_spec, F=FREQ_MASK_PARAM, num_masks=FREQ_MASK_NUM):
    """Frequency masking (SpecAugment)"""
    cloned = mel_spec.copy()
    num_mel_channels = cloned.shape[1]
    
    for _ in range(num_masks):
        f = np.random.randint(0, F)
        f_zero = np.random.randint(0, num_mel_channels - f)
        cloned[:, f_zero:f_zero + f, :] = 0
    
    return cloned


def time_mask(mel_spec, T=TIME_MASK_PARAM, num_masks=TIME_MASK_NUM):
    """Time masking (SpecAugment)"""
    cloned = mel_spec.copy()
    time_steps = cloned.shape[2]
    
    for _ in range(num_masks):
        t = np.random.randint(0, T)
        t_zero = np.random.randint(0, time_steps - t)
        cloned[:, :, t_zero:t_zero + t] = 0
    
    return cloned


def cutout(mel_spec, n_holes=CUTOUT_N_HOLES, length=CUTOUT_LENGTH):
    """Cutout augmentation"""
    h = mel_spec.shape[1]
    w = mel_spec.shape[2]
    
    mask = np.ones((h, w), np.float32)
    
    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        
        mask[y1: y2, x1: x2] = 0.
    
    mask = mask[np.newaxis, ...]
    return mel_spec * mask


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=CUTMIX_ALPHA):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get random box
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    
    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match box ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


# ============================================================
# Audio-level Data Augmentation (Time Stretch & Pitch Shift)
# ============================================================

def augment_audio_dataset(csv_path, audio_dir, output_dir, augment_types=['original', 'stretch', 'pitch_shift'],
                          minority_classes=[1, 6], heavy_augment_minority=True):
    """
    对 UrbanSound8K 数据集进行音频增强，对少数类进行更丰富的增强以平衡类别分布
    
    增强策略：
    - 多数类：Original + Time Stretch (2.0x) + Pitch Shift (+1 semitone)  [3x]
    - 少数类：Original + 多种 Time Stretch (0.8x, 1.2x, 1.5x, 2.0x) + 多种 Pitch Shift (-2, -1, +1, +2)  [9x]
    
    Args:
        csv_path: CSV metadata文件路径
        audio_dir: 原始音频目录
        output_dir: 增强后音频保存目录
        augment_types: 要生成的增强类型列表（用于向后兼容，如果 heavy_augment_minority=False）
        minority_classes: 少数类的 classID 列表（默认: [1, 6] 对应 car_horn 和 gun_shot）
        heavy_augment_minority: 是否对少数类应用更丰富的增强策略
    
    Returns:
        DataFrame with augmented audio paths
    """
    print("\n" + "=" * 80)
    print("Audio Data Augmentation with Class-Specific Strategies")
    print("=" * 80)
    
    # Class names for display
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    
    # Read metadata
    df = pd.read_csv(csv_path)
    
    # Display class distribution
    print("\nOriginal class distribution:")
    class_counts = df['classID'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        marker = " 🎯 (MINORITY - Heavy Augmentation)" if class_id in minority_classes else ""
        print(f"  Class {class_id} ({class_names[class_id]:18s}): {count:4d} samples{marker}")
    
    if heavy_augment_minority:
        print(f"\nAugmentation strategy:")
        print(f"  Majority classes: Original + Stretch(2.0x) + PitchShift(+1)  → 3x multiplier")
        print(f"  Minority classes: Original + Stretch(0.8x,1.2x,1.5x,2.0x) + PitchShift(-2,-1,+1,+2)  → 9x multiplier")
    else:
        print(f"\nAugmentation strategy: Uniform for all classes")
        print(f"  All classes: {augment_types}  → {len(augment_types)}x multiplier")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    augmented_records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting audio"):
        fold = row['fold']
        filename = row['slice_file_name']
        class_id = row['classID']
        
        # Input audio path
        audio_path = Path(audio_dir) / f"fold{fold}" / filename
        
        # Create fold directory in output
        fold_output_dir = output_dir / f"fold{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine augmentation strategy based on class
        is_minority = class_id in minority_classes
        
        # Load original audio
        try:
            y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
            
            # Always save original
            original_filename = f"{filename[:-4]}_original.wav"
            original_path = fold_output_dir / original_filename
            sf.write(str(original_path), y, sr)
            
            augmented_records.append({
                'fold': fold,
                'slice_file_name': original_filename,
                'classID': class_id,
                'augmentation': 'original',
                'class': class_names[class_id]
            })
            
            # Apply augmentation based on strategy
            if heavy_augment_minority and is_minority:
                # Heavy augmentation for minority classes
                # Multiple time stretch rates
                stretch_rates = [0.8, 1.2, 1.5, 2.0]
                for rate in stretch_rates:
                    y_stretch = librosa.effects.time_stretch(y, rate=rate)
                    stretch_filename = f"{filename[:-4]}_stretch_{rate:.1f}x.wav"
                    stretch_path = fold_output_dir / stretch_filename
                    sf.write(str(stretch_path), y_stretch, sr)
                    
                    augmented_records.append({
                        'fold': fold,
                        'slice_file_name': stretch_filename,
                        'classID': class_id,
                        'augmentation': f'time_stretch_{rate:.1f}x',
                        'class': class_names[class_id]
                    })
                
                # Multiple pitch shifts
                pitch_shifts = [-2, -1, +1, +2]
                for n_steps in pitch_shifts:
                    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                    shift_filename = f"{filename[:-4]}_pitch_{n_steps:+d}st.wav"
                    shift_path = fold_output_dir / shift_filename
                    sf.write(str(shift_path), y_shift, sr)
                    
                    augmented_records.append({
                        'fold': fold,
                        'slice_file_name': shift_filename,
                        'classID': class_id,
                        'augmentation': f'pitch_shift_{n_steps:+d}',
                        'class': class_names[class_id]
                    })
            else:
                # Standard augmentation for majority classes
                # Time Stretch (2.0x)
                if 'stretch' in augment_types:
                    y_stretch = librosa.effects.time_stretch(y, rate=2.0)
                    stretch_filename = f"{filename[:-4]}_stretch.wav"
                    stretch_path = fold_output_dir / stretch_filename
                    sf.write(str(stretch_path), y_stretch, sr)
                    
                    augmented_records.append({
                        'fold': fold,
                        'slice_file_name': stretch_filename,
                        'classID': class_id,
                        'augmentation': 'time_stretch_2.0x',
                        'class': class_names[class_id]
                    })
                
                # Pitch Shift (+1 semitone)
                if 'pitch_shift' in augment_types:
                    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
                    shift_filename = f"{filename[:-4]}_pitch_shift.wav"
                    shift_path = fold_output_dir / shift_filename
                    sf.write(str(shift_path), y_shift, sr)
                    
                    augmented_records.append({
                        'fold': fold,
                        'slice_file_name': shift_filename,
                        'classID': class_id,
                        'augmentation': 'pitch_shift_+1',
                        'class': class_names[class_id]
                    })
                
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue
    
    # Create augmented DataFrame
    augmented_df = pd.DataFrame(augmented_records)
    
    print(f"\n" + "=" * 80)
    print("Audio Augmentation Complete")
    print("=" * 80)
    print(f"Original samples: {len(df)}")
    print(f"Augmented samples: {len(augmented_df)}")
    print(f"Overall multiplier: {len(augmented_df) / len(df):.1f}x")
    
    # Display new class distribution
    print("\nNew class distribution after augmentation:")
    new_class_counts = augmented_df['classID'].value_counts().sort_index()
    for class_id, count in new_class_counts.items():
        original_count = class_counts.get(class_id, 0)
        multiplier = count / original_count if original_count > 0 else 0
        marker = " 🎯" if class_id in minority_classes else ""
        print(f"  Class {class_id} ({class_names[class_id]:18s}): {count:5d} samples (×{multiplier:.1f}){marker}")
    
    return augmented_df


def augment_single_audio(audio_path, output_folder, augment_types=['original', 'stretch', 'pitch_shift'], sr=SAMPLE_RATE):
    """
    对单个音频文件应用增强
    
    Args:
        audio_path: 输入音频文件路径
        output_folder: 输出文件夹
        augment_types: 要生成的增强类型列表
        sr: 采样率
    
    Returns:
        List of augmented file paths
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    y, file_sr = librosa.load(str(audio_path), sr=sr)
    
    audio_path = Path(audio_path)
    base_name = audio_path.stem
    
    augmented_paths = []
    
    # 1. Original
    if 'original' in augment_types:
        original_path = output_folder / f"{base_name}_original.wav"
        sf.write(str(original_path), y, file_sr)
        augmented_paths.append(str(original_path))
        print(f"Saved: {original_path}")
    
    # 2. Time Stretch (2x speed)
    if 'stretch' in augment_types:
        y_stretch = librosa.effects.time_stretch(y, rate=2.0)
        stretch_path = output_folder / f"{base_name}_stretch.wav"
        sf.write(str(stretch_path), y_stretch, file_sr)
        augmented_paths.append(str(stretch_path))
        print(f"Saved: {stretch_path} (Time Stretch x2)")
    
    # 3. Pitch Shift (+1 semitone)
    if 'pitch_shift' in augment_types:
        y_shift = librosa.effects.pitch_shift(y, sr=file_sr, n_steps=1)
        shift_path = output_folder / f"{base_name}_pitch_shift.wav"
        sf.write(str(shift_path), y_shift, file_sr)
        augmented_paths.append(str(shift_path))
        print(f"Saved: {shift_path} (Pitch Shift +1 semitone)")
    
    return augmented_paths


# ============================================================
# Dataset Classes
# ============================================================

class CachedMelDataset(Dataset):
    """
    Dataset for UrbanSound8K that loads cached mel spectrograms
    
    Features:
    - Lazy loading of pre-computed mel spectrograms from .npy files
    - Training mode: applies data augmentation (SpecAugment, Gaussian noise, etc.)
    - Validation/Test mode: no augmentation for fair evaluation
    - Optionally returns salience info for sample weighting
    """
    
    def __init__(self, df, mode='train', augment=True, return_salience=False):
        """
        Args:
            df: DataFrame with columns ['cached_path', 'classID']
            mode: 'train' or 'val' - determines if augmentation is applied
            augment: Whether to apply augmentation (only effective in train mode)
            return_salience: Whether to return salience info (for sample weighting)
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.return_salience = return_salience
        self.has_salience = 'salience' in df.columns
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Lazy load cached mel spectrogram
        feature = np.load(row['cached_path'])
        
        # Apply augmentation ONLY in training mode
        if self.augment:
            # Random augmentation selection
            if np.random.rand() < AUG_PROBABILITY:
                aug_type = np.random.choice(['noise', 'freq_mask', 'time_mask', 'cutout'])
                
                if aug_type == 'noise':
                    feature = add_gaussian_noise(feature)
                elif aug_type == 'freq_mask':
                    feature = freq_mask(feature, F=FREQ_MASK_PARAM)
                elif aug_type == 'time_mask':
                    feature = time_mask(feature, T=TIME_MASK_PARAM)
                elif aug_type == 'cutout':
                    feature = cutout(feature)
        
        # Convert to tensor
        feature = torch.from_numpy(feature).float()
        label = torch.tensor(row['classID'], dtype=torch.long)
        
        # Return salience if requested and available
        if self.return_salience and self.has_salience:
            salience = torch.tensor(row['salience'], dtype=torch.long)
            return feature, label, salience
        
        return feature, label


class UrbanSound8KDataset(Dataset):
    """
    Dataset for UrbanSound8K with on-the-fly feature extraction (no caching)
    
    Features:
    - Loads audio files and converts to mel spectrogram on-the-fly
    - Training mode: applies data augmentation
    - Validation/Test mode: no augmentation
    
    Use this when you don't want to pre-cache features.
    """
    
    def __init__(self, df, audio_dir, mode='train', augment=True, use_pcen=USE_PCEN):
        """
        Args:
            df: DataFrame with columns ['fold', 'slice_file_name', 'classID']
            audio_dir: Path to audio directory
            mode: 'train' or 'val'
            augment: Whether to apply augmentation
            use_pcen: If True, use PCEN instead of Log-Mel
        """
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.use_pcen = use_pcen
        self.target_length = int(SAMPLE_RATE * DURATION)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fold = row['fold']
        filename = row['slice_file_name']
        
        # Load audio and convert to mel spectrogram
        audio_path = self.audio_dir / f"fold{fold}" / filename
        audio = load_audio_with_circular_padding(audio_path, self.target_length)
        feature = audio_to_mel_spectrogram(audio, use_pcen=self.use_pcen)
        
        # Apply augmentation ONLY in training mode
        if self.augment:
            if np.random.rand() < AUG_PROBABILITY:
                aug_type = np.random.choice(['noise', 'freq_mask', 'time_mask', 'cutout'])
                
                if aug_type == 'noise':
                    feature = add_gaussian_noise(feature)
                elif aug_type == 'freq_mask':
                    feature = freq_mask(feature, F=FREQ_MASK_PARAM)
                elif aug_type == 'time_mask':
                    feature = time_mask(feature, T=TIME_MASK_PARAM)
                elif aug_type == 'cutout':
                    feature = cutout(feature)
        
        # Convert to tensor
        feature = torch.from_numpy(feature).float()
        label = torch.tensor(row['classID'], dtype=torch.long)
        
        return feature, label

