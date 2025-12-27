#!/usr/bin/env python3
"""
Inference script for ResLiteAudioCNN and FusedCNN to generate submission file
Supports both Mel spectrograms and GFCC+STE fusion features
"""
import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.fftpack import dct

# Import config and model
from config import *
from model import ResLiteAudioCNN, FusedCNN


# ============================================================
# Helper Functions
# ============================================================

def log(mel_spec, eps=1e-10):
    """Convert mel spectrogram to log scale"""
    return np.log10(mel_spec + eps)


# ============================================================
# Feature Extraction for Fusion Features - COMMENTED OUT
# ============================================================

# def extract_gfcc(y, sr, n_filters=None, n_ceps=None, win_time=None, hop_time=None, eps=1e-10):
#     """Extract GFCC features"""
#     if n_filters is None:
#         n_filters = GFCC_N_FILTERS
#     if n_ceps is None:
#         n_ceps = GFCC_N_CEPS
#     if win_time is None:
#         win_time = GFCC_WIN_TIME
#     if hop_time is None:
#         hop_time = GFCC_HOP_TIME
#     
#     from gammatone.gtgram import gtgram
#     gspec = gtgram(y, sr, win_time, hop_time, n_filters, 50)
#     gspec = np.maximum(gspec, eps)
#     log_gspec = np.log(gspec)
#     gfcc = dct(log_gspec, type=2, axis=0, norm='ortho')[:n_ceps, :]
#     return gfcc.astype(np.float32)


# def extract_ste(y, sr, frame_length=None, hop_length=None, eps=1e-10):
#     """Extract short-time energy"""
#     if frame_length is None:
#         frame_length = STE_FRAME_LENGTH
#     if hop_length is None:
#         hop_length = STE_HOP_LENGTH
#     
#     rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)
#     ste = np.maximum(rms**2, eps)
#     return ste.astype(np.float32)


# def fuse_gfcc_ste(gfcc, ste):
#     """Fuse GFCC and STE features"""
#     T = min(gfcc.shape[1], ste.shape[1])
#     fused = np.concatenate([gfcc[:, :T], ste[:, :T]], axis=0)
#     return fused


# def normalize_feature(feature, mean, std):
#     """Apply feature-wise normalization"""
#     if feature.ndim == 3:
#         # (1, F, T) format
#         mean = mean[np.newaxis, :, np.newaxis]
#         std = std[np.newaxis, :, np.newaxis]
#     else:
#         # (F, T) format
#         mean = mean[:, np.newaxis]
#         std = std[:, np.newaxis]
#     
#     return (feature - mean) / std


# def extract_fused_feature(wav_path, sr=None, duration=None, n_filters=None, 
#                           n_ceps=None, win_time=None, hop_time=None):
#     """Extract GFCC+STE fusion feature"""
#     if sr is None:
#         sr = SAMPLE_RATE
#     if duration is None:
#         duration = DURATION
#     if n_filters is None:
#         n_filters = GFCC_N_FILTERS
#     if n_ceps is None:
#         n_ceps = GFCC_N_CEPS
#     if win_time is None:
#         win_time = GFCC_WIN_TIME
#     if hop_time is None:
#         hop_time = GFCC_HOP_TIME
#     
#     y, _ = librosa.load(wav_path, sr=sr, mono=True)
#     
#     # Pad or trim to target length
#     target_len = int(sr * duration)
#     if len(y) < target_len:
#         reps = int(np.ceil(target_len / len(y)))
#         y = np.tile(y, reps)[:target_len]
#     else:
#         y = y[:target_len]
#     
#     gfcc = extract_gfcc(y, sr, n_filters=n_filters, n_ceps=n_ceps, 
#                        win_time=win_time, hop_time=hop_time)
#     
#     hop_length = int(round(hop_time * sr))
#     frame_length = int(round(win_time * sr))
#     ste = extract_ste(y, sr, frame_length=frame_length, hop_length=hop_length)
#     
#     fused = fuse_gfcc_ste(gfcc, ste)
#     fused = fused[np.newaxis, ...]  # Add channel dimension
#     
#     return fused.astype(np.float32)


# ============================================================
# Test Dataset
# ============================================================

def load_audio_with_circular_padding(audio_path, target_length, sr=SAMPLE_RATE):
    """Load audio and apply circular padding if too short"""
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    
    current_length = len(y)
    
    if current_length < target_length:
        # Circular padding (repeat the audio)
        num_repeats = (target_length // current_length) + 1
        y = np.tile(y, num_repeats)
        y = y[:target_length]
    elif current_length > target_length:
        # Center crop
        start = (current_length - target_length) // 2
        y = y[start:start + target_length]
    
    return y


# HPSS Feature Extraction - COMMENTED OUT (Use single channel instead)
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
#                                                n_fft=n_fft, hop_length=hop_length, fmax=FMAX)
#     mel_h = librosa.feature.melspectrogram(S=np.abs(H)**2, sr=sr, n_mels=n_mels, 
#                                            n_fft=n_fft, hop_length=hop_length, fmax=FMAX)
#     mel_p = librosa.feature.melspectrogram(S=np.abs(P)**2, sr=sr, n_mels=n_mels, 
#                                            n_fft=n_fft, hop_length=hop_length, fmax=FMAX)
#     
#     # Apply log transformation and stack
#     X = np.stack([log(mel_full), log(mel_h), log(mel_p)], axis=0)
#     
#     return X


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
        fmax=FMAX
    )
    
    # Apply PCEN or Log-Mel transformation
    if use_pcen:
        # PCEN (Per-Channel Energy Normalization)
        mel_spec_processed = librosa.pcen(
            mel_spec * (2**31),
            sr=sr,
            hop_length=hop_length,
            gain=PCEN_GAIN,
            bias=PCEN_BIAS,
            power=PCEN_POWER,
            time_constant=PCEN_TIME_CONSTANT,
            eps=PCEN_EPS
        )
    else:
        # Log-Mel (traditional approach)
        mel_spec_processed = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Add channel dimension: (1, n_mels, time)
    mel_spec_processed = mel_spec_processed[np.newaxis, ...]
    
    return mel_spec_processed.astype(np.float32)


class TestDataset(Dataset):
    """Dataset for test set - supports both Mel and Fusion features (Fusion DISABLED)"""
    
    def __init__(self, df, audio_dir, use_fusion=False, use_pcen=USE_PCEN, norm_stats=None):
        """
        Args:
            df: DataFrame with 'ID' and 'slice_file_name' columns (in correct order)
            audio_dir: Path to test audio directory
            use_fusion: If True, extract GFCC+STE fusion features (DISABLED)
            use_pcen: If True, use PCEN instead of Log-Mel (only for mel mode)
            norm_stats: Normalization statistics for fusion features (DISABLED)
        """
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.target_length = int(SAMPLE_RATE * DURATION)
        # self.use_fusion = use_fusion  # Fusion disabled
        self.use_pcen = use_pcen
        # self.norm_stats = norm_stats  # Fusion disabled
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row['ID']
        filename = row['slice_file_name']
        
        audio_path = self.audio_dir / "test" / filename
        
        # if self.use_fusion:
        #     # Extract GFCC+STE fusion features
        #     feature = extract_fused_feature(str(audio_path))
        #     
        #     # Apply normalization if provided
        #     if self.norm_stats is not None:
        #         feature = normalize_feature(
        #             feature,
        #             self.norm_stats['mean'],
        #             self.norm_stats['std']
        #         )
        # else:
        # Extract single-channel Mel spectrogram
        audio = load_audio_with_circular_padding(audio_path, self.target_length)
        feature = audio_to_mel_spectrogram(audio, use_pcen=self.use_pcen)
        
        # Convert to tensor
        feature = torch.from_numpy(feature).float()
        
        return feature, file_id


# ============================================================
# Inference Functions
# ============================================================

def load_model(checkpoint_path, device, num_classes=10, use_fusion=False):
    """Load trained model from checkpoint (fusion DISABLED)"""
    # if use_fusion:
    #     model = FusedCNN(num_classes=num_classes, dropout=0.3)
    # else:
    model = ResLiteAudioCNN(num_classes=num_classes, dropout=0.3)  # Fusion disabled
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def predict_with_tta(model, data, device, n_tta=5):
    """
    Predict with Test Time Augmentation
    
    Args:
        model: Trained model
        data: Input data tensor
        device: torch device
        n_tta: Number of TTA iterations
    
    Returns:
        Average prediction
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original prediction
        output = model(data.to(device))
        predictions.append(torch.softmax(output, dim=1).cpu())
        
        # TTA with horizontal flip (not applicable for audio, so we use different crops)
        # For audio, we can use small time shifts
        for _ in range(n_tta - 1):
            # Small random crop/shift (simulate variation)
            output = model(data.to(device))
            predictions.append(torch.softmax(output, dim=1).cpu())
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    
    return avg_pred


def generate_submission(model, test_loader, device, use_tta=True, n_tta=5):
    """
    Generate predictions on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: torch device
        use_tta: Whether to use Test Time Augmentation
        n_tta: Number of TTA iterations
    
    Returns:
        DataFrame with predictions
    """
    model.eval()
    all_ids = []
    all_predictions = []
    
    with torch.no_grad():
        for data, file_ids in tqdm(test_loader, desc='Predicting'):
            data = data.to(device)
            
            if use_tta:
                # Use TTA
                outputs = []
                for _ in range(n_tta):
                    output = model(data)
                    outputs.append(torch.softmax(output, dim=1))
                
                # Average predictions
                output = torch.stack(outputs).mean(dim=0)
            else:
                # Single prediction
                output = model(data)
                output = torch.softmax(output, dim=1)
            
            # Get predicted class
            preds = output.argmax(dim=1).cpu().numpy()
            
            all_ids.extend(file_ids.cpu().numpy())
            all_predictions.extend(preds)
    
    # Create submission DataFrame - IDs should already be in order
    submission_df = pd.DataFrame({
        'ID': all_ids,
        'TARGET': all_predictions
    })
    
    # Sort by ID to ensure correct order (should already be sorted, but just to be safe)
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)
    
    return submission_df


def ensemble_predictions(checkpoint_paths, test_loader, device, use_tta=True, use_fusion=False):
    """
    Ensemble predictions from multiple models (fusion DISABLED)
    
    Args:
        checkpoint_paths: List of checkpoint paths
        test_loader: Test data loader
        device: torch device
        use_tta: Whether to use TTA
        use_fusion: Whether models use fusion features (DISABLED)
    
    Returns:
        DataFrame with ensemble predictions
    """
    # model_type = "FusedCNN" if use_fusion else "ResLiteAudioCNN"  # Fusion disabled
    model_type = "ResLiteAudioCNN"
    print(f"\nEnsembling {len(checkpoint_paths)} {model_type} models...")
    
    all_probs = []
    file_ids = None
    
    for checkpoint_path in checkpoint_paths:
        print(f"\nLoading model: {checkpoint_path}")
        model = load_model(checkpoint_path, device, use_fusion=False)  # Fusion disabled
        
        model.eval()
        batch_probs = []
        batch_ids = []
        
        with torch.no_grad():
            for data, ids in tqdm(test_loader, desc=f'Predicting'):
                data = data.to(device)
                
                # ===== ABLATION: TTA DISABLED =====
                # if use_tta:
                #     # Use TTA
                #     outputs = []
                #     for _ in range(5):
                #         output = model(data)
                #         outputs.append(torch.softmax(output, dim=1))
                #     output = torch.stack(outputs).mean(dim=0)
                # else:
                #     output = model(data)
                #     output = torch.softmax(output, dim=1)
                
                # Ablation: Direct inference without TTA
                output = model(data)
                output = torch.softmax(output, dim=1)
                # ===== END ABLATION =====
                
                batch_probs.append(output.cpu().numpy())
                batch_ids.extend(ids.cpu().numpy())
        
        # Concatenate all batches
        probs = np.vstack(batch_probs)
        all_probs.append(probs)
        
        if file_ids is None:
            file_ids = batch_ids
    
    # Average probabilities from all models
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'ID': file_ids,
        'TARGET': ensemble_preds
    })
    
    # Sort by ID to ensure correct order (should already be sorted)
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)
    
    return submission_df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate submission file with ResLiteAudioCNN or FusedCNN')
    parser.add_argument('--test-csv', type=str, default='Kaggle_Data/metadata/kaggle_test.csv',
                       help='Path to test CSV')
    parser.add_argument('--audio-dir', type=str, default='Kaggle_Data/audio',
                       help='Path to audio directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (or comma-separated list for ensemble)')
    parser.add_argument('--output', type=str, default='submission_reslite.csv',
                       help='Output submission file path')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID to use')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--tta', action='store_true',
                       help='Use Test Time Augmentation')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble of multiple models')
    # parser.add_argument('--use-fusion', action='store_true',
    #                    help='Use GFCC+STE fusion features (FusedCNN model)')  # COMMENTED OUT
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Display feature type (fusion disabled)
    # if args.use_fusion:
    #     print(f"\nFeature type: GFCC+STE Fusion")
    #     print(f"  Model: FusedCNN")
    #     print(f"  GFCC parameters:")
    #     print(f"    - n_filters: {GFCC_N_FILTERS}")
    #     print(f"    - n_ceps: {GFCC_N_CEPS}")
    #     print(f"    - win_time: {GFCC_WIN_TIME}s")
    #     print(f"    - hop_time: {GFCC_HOP_TIME}s")
    # else:
    print(f"\nFeature type: {'PCEN' if USE_PCEN else 'Log-Mel'} Spectrogram")
    print(f"  Model: ResLiteAudioCNN")
    if USE_PCEN:
        print(f"  PCEN parameters:")
        print(f"    - gain: {PCEN_GAIN}")
        print(f"    - bias: {PCEN_BIAS}")
        print(f"    - power: {PCEN_POWER}")
        print(f"    - time_constant: {PCEN_TIME_CONSTANT}")
    
    # Load normalization statistics for fusion features - COMMENTED OUT
    norm_stats = None
    # if args.use_fusion and USE_FEATURE_NORMALIZATION:
    #     norm_stats_path = Path(NORM_STATS_PATH)
    #     if norm_stats_path.exists():
    #         print(f"\n✓ Loading normalization stats from: {norm_stats_path}")
    #         stats = np.load(norm_stats_path)
    #         norm_stats = {'mean': stats['mean'], 'std': stats['std']}
    #         print(f"  Mean shape: {norm_stats['mean'].shape}")
    #         print(f"  Std shape: {norm_stats['std'].shape}")
    #     else:
    #         print(f"\n⚠ Warning: Normalization stats not found at {norm_stats_path}")
    #         print("  Proceeding without normalization (may affect accuracy)")
    
    # Read test CSV
    test_df = pd.read_csv(args.test_csv)
    print(f"\nTest samples: {len(test_df)}")
    
    # Create test dataset and dataloader (fusion disabled)
    test_dataset = TestDataset(test_df, args.audio_dir, 
                               use_fusion=False,  # Fusion disabled
                               use_pcen=USE_PCEN,
                               norm_stats=None)  # Normalization disabled
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # MUST be False to maintain order!
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Generate predictions (fusion disabled)
    if args.checkpoint is None:
        # Auto-detect checkpoints based on feature type (fusion disabled)
        # if args.use_fusion:
        #     # Look for fusion model checkpoints
        #     checkpoint_patterns = [
        #         "outputs_fusion/fused_valfolds*_best.pth",
        #         "outputs_fusion/fused_fold*_best.pth"
        #     ]
        # else:
        # Look for reslite model checkpoints (single-channel models)
        # Supports both fold-based (e.g., reslite_valfolds_8_best.pth) 
        # and random split models (e.g., reslite_valfolds_random_0.125_seed42_best.pth)
        checkpoint_patterns = [
            "weights/reslite_valfoldsrandom*_best.pth"
        ]
        
        checkpoint_paths = []
        for pattern in checkpoint_patterns:
            from glob import glob
            checkpoint_paths.extend(glob(pattern))
        
        if len(checkpoint_paths) == 0:
            print(f"Error: No checkpoint files found!")
            print(f"  Searched patterns: {checkpoint_patterns}")
            return
        
        print(f"\nAuto-detected {len(checkpoint_paths)} model(s) for {'ensemble' if len(checkpoint_paths) > 1 else 'inference'}:")
        for path in checkpoint_paths:
            print(f"  - {path}")
        
        # Use ensemble by default (fusion disabled)
        submission_df = ensemble_predictions(
            checkpoint_paths, test_loader, device, 
            use_tta=args.tta, use_fusion=False  # Fusion disabled
        )
    elif args.ensemble:
        # Ensemble mode with specified checkpoints (fusion disabled)
        checkpoint_paths = args.checkpoint.split(',')
        
        print(f"\nUsing ensemble of {len(checkpoint_paths)} models:")
        for path in checkpoint_paths:
            print(f"  - {path}")
        
        submission_df = ensemble_predictions(
            checkpoint_paths, test_loader, device, 
            use_tta=args.tta, use_fusion=False  # Fusion disabled
        )
    else:
        # Single model mode (fusion disabled)
        checkpoint_path = args.checkpoint
        
        print(f"\nUsing single model: {checkpoint_path}")
        model = load_model(checkpoint_path, device, use_fusion=False)  # Fusion disabled
        
        submission_df = generate_submission(
            model, test_loader, device, use_tta=args.tta
        )
    
    # Save submission
    submission_df.to_csv(args.output, index=False)
    
    print("\n" + "=" * 80)
    print("Submission File Generated!")
    print("=" * 80)
    print(f"Output file: {args.output}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Format: ID,TARGET")
    print(f"\nClass distribution:")
    print(submission_df['TARGET'].value_counts().sort_index())
    
    # Preview
    print(f"\nPreview:")
    print(submission_df.head(10))
    
    print("\n✓ Ready to submit to Kaggle!")


if __name__ == '__main__':
    main()

