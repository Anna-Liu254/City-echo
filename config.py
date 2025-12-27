# ============================================================
# Audio Parameters
# ============================================================
SAMPLE_RATE = 22050  # Target sample rate
DURATION = 4  # seconds - standardize all audio to 4 seconds
N_MELS = 128  # Number of mel frequency bins
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT
FMAX = 8000  # Maximum frequency for mel spectrogram

# ============================================================
# Training Parameters
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 2e-3  # Starting learning rate for OneCycleLR
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8  # Number of data loading workers
NUM_CLASSES = 10

# ============================================================
# Data Augmentation Parameters
# ============================================================
# Basic augmentation
AUG_PROBABILITY = 0.7  # Probability to apply augmentation

# Gaussian Noise
NOISE_LEVEL = 0.003  # Gaussian noise level

# SpecAugment
FREQ_MASK_PARAM = 20  # Frequency masking parameter (width)
TIME_MASK_PARAM = 40  # Time masking parameter (width)
FREQ_MASK_NUM = 2  # Number of frequency masks
TIME_MASK_NUM = 2  # Number of time masks

# Cutout
CUTOUT_N_HOLES = 3  # Number of cutout holes
CUTOUT_LENGTH = 20  # Length of cutout holes

# Mixup (applied at batch level)
MIXUP_ALPHA = 0.2  # Mixup alpha parameter
USE_MIXUP = True  # Whether to use Mixup
MIXUP_PROB = 0.9  # Probability to apply Mixup

# CutMix (applied at batch level)
CUTMIX_ALPHA = 1.0  # CutMix alpha parameter
USE_CUTMIX = False  # Whether to use CutMix
CUTMIX_PROB = 0.5  # Probability to apply CutMix

# ============================================================
# PCEN (Per-Channel Energy Normalization) Settings
# ============================================================
USE_PCEN = False  # Use PCEN instead of Log-Mel (set to True for PCEN model)

# PCEN hyperparameters (only used when USE_PCEN=True)
PCEN_GAIN = 0.6          # AGC strength, range: [0.6, 0.98]
PCEN_BIAS = 2             # Prevents division by zero, range: [2, 10]
PCEN_POWER = 0.5          # Compression exponent, range: [0.25, 0.5]
PCEN_TIME_CONSTANT = 0.800  # AGC time constant (400ms for urban noise)
PCEN_EPS = 1e-6           # Small constant for numerical stability

# ============================================================
# Multi-Feature Fusion Settings (GFCC + STE) - COMMENTED OUT
# ============================================================
# USE_FUSION_FEATURES = True  # Set to True to use GFCC+STE fusion features
# GFCC_N_FILTERS = 64  # Number of gammatone filters
# GFCC_N_CEPS = 20  # Number of cepstral coefficients
# GFCC_WIN_TIME = 0.040  # Window size in seconds (25ms)
# GFCC_HOP_TIME = 0.010  # Hop size in seconds (10ms)
# STE_FRAME_LENGTH = int(round(GFCC_WIN_TIME * SAMPLE_RATE))  # Frame length for short-time energy
# STE_HOP_LENGTH = int(round(GFCC_HOP_TIME * SAMPLE_RATE))  # Hop length for STE (should match GFCC hop)

# Feature normalization settings
# USE_FEATURE_NORMALIZATION = True  # Apply feature-wise normalization (mean/std)
# NORM_STATS_PATH = 'cache/fusion_norm_stats.npz'  # Path to save/load normalization statistics

# ============================================================
# Salience-based Sample Weighting (Data Cleaning)
# ============================================================
USE_SALIENCE_WEIGHTING = False  # Use salience info for sample weighting
BACKGROUND_WEIGHT = 0.5  # Weight for background samples (salience=2), range: [0.0, 1.0]
                         # Lower value = less influence from noisy/background samples
                         # 1.0 = no weighting (treat all samples equally)
                         # 0.5 = background samples contribute half as much to loss
                         # 0.0 = completely ignore background samples

# ============================================================
# Other Settings
# ============================================================
USE_CLASS_WEIGHTS = False  # Disable class weights