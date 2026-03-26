"""
 preprocessing.py - Data Cleaning & Feature Engineering

 Phase 3: Preprocessing Pipeline
 - Data cleaning & merge
 - Feature engineering (cyclical encoding, lag features, rolling stats)
 - Time-based split
 - Sequence creation for LSTM
 - Building ID embedding preparation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# Config
SEED = 42
np.random.seed(SEED)

# Constants - ASHRAE train data is 2016 only!
# Split: Train (Jan-Oct), Val (Nov), Test (Dec)
TRAIN_END = '2016-10-31'
VAL_END = '2016-11-30'
SEQ_LENGTH = 168  # 7 days
TARGET_HORIZON = 24  # 24 hours ahead
OUTLIER_BUILDINGS = [1099]

# Memory optimization: sample size (None = use all data, or set number like 500000)
SAMPLE_SIZE = None  # Set to e.g., 1000000 for memory-constrained systems

# Sequence stride - skip timesteps to reduce sequence count
SEQUENCE_STRIDE = 24  # Create sequence every 24 hours instead of every hour

# Max buildings to use (None = all, or set number)
MAX_BUILDINGS = 200  # Limit buildings for memory efficiency


def optimize_dtypes(df):
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            if df[col].min() >= 0 and df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            elif df[col].min() >= -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
        elif col_type == 'object':
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')
    return df


def load_and_clean_data(data_path, sample_size=None):
    """
    Load ASHRAE data and perform initial cleaning
    Memory-optimized version
    """
    print("=== LOADING & CLEANING DATA ===")

    # Load with optimized dtypes
    dtype_train = {
        'building_id': 'int16',
        'meter': 'int8',
        'meter_reading': 'float32'
    }
    dtype_weather = {
        'site_id': 'int8',
        'air_temperature': 'float32',
        'cloud_coverage': 'float32',
        'dew_temperature': 'float32',
        'precip_depth_1_hr': 'float32',
        'sea_level_pressure': 'float32',
        'wind_direction': 'float32',
        'wind_speed': 'float32'
    }

    train = pd.read_csv(Path(data_path) / 'train.csv', dtype=dtype_train)
    weather_train = pd.read_csv(Path(data_path) / 'weather_train.csv', dtype=dtype_weather)
    building_meta = pd.read_csv(Path(data_path) / 'building_metadata.csv')

    print(f"Loaded: train={train.shape}, weather={weather_train.shape}, meta={building_meta.shape}")

    # Filter electricity only (meter=0) - in-place to save memory
    train = train[train['meter'] == 0]
    train.drop(columns=['meter'], inplace=True)
    print(f"Filtered to electricity: {len(train):,} rows")

    # Remove outlier buildings
    train = train[~train['building_id'].isin(OUTLIER_BUILDINGS)]
    print(f"After removing outliers: {len(train):,} rows")

    # Convert timestamps FIRST (before any year-based operations)
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])
    
    # Check data range
    print(f"Data time range: {train['timestamp'].min()} to {train['timestamp'].max()}")

    # Sample by BUILDINGS (not rows) to keep temporal continuity
    if MAX_BUILDINGS is not None:
        unique_buildings = train['building_id'].unique()
        if len(unique_buildings) > MAX_BUILDINGS:
            print(f"⚠️ Sampling {MAX_BUILDINGS} buildings (from {len(unique_buildings)}) for memory efficiency...")
            
            # Select buildings with most data points for better coverage
            building_counts = train.groupby('building_id').size().sort_values(ascending=False)
            top_buildings = building_counts.head(MAX_BUILDINGS).index.tolist()
            
            np.random.seed(42)
            sampled_buildings = top_buildings  # Use top buildings by data count
            train = train[train['building_id'].isin(sampled_buildings)]
            print(f"After sampling: {len(train):,} rows ({len(sampled_buildings)} buildings)")

    # Optimize memory
    building_meta = optimize_dtypes(building_meta)

    gc.collect()

    return train, weather_train, building_meta


def merge_data(train_df, weather_df, building_meta):
    """
    Merge train, weather, and building metadata
    Memory-optimized version
    """
    print("\n=== MERGING DATA ===")

    # Keep only needed columns from building_meta
    meta_cols = ['building_id', 'site_id', 'square_feet', 'year_built', 'primary_use']
    building_meta_slim = building_meta[meta_cols].copy()

    # Merge with building metadata
    train_df = train_df.merge(building_meta_slim, on='building_id', how='left')
    print(f"After merging building metadata: {len(train_df):,} rows")
    del building_meta_slim
    gc.collect()

    # Keep only needed weather columns
    weather_cols_keep = ['site_id', 'timestamp', 'air_temperature', 'dew_temperature', 'wind_speed']
    weather_df_slim = weather_df[weather_cols_keep].copy()

    # Merge with weather
    train_df = train_df.merge(
        weather_df_slim,
        on=['site_id', 'timestamp'],
        how='left'
    )
    print(f"After merging weather: {len(train_df):,} rows")
    del weather_df_slim
    gc.collect()

    # Forward fill missing weather data per site
    print("Filling missing weather data...")
    train_df.sort_values(['site_id', 'timestamp'], inplace=True)

    weather_cols = ['air_temperature', 'dew_temperature', 'wind_speed']
    for col in weather_cols:
        if col in train_df.columns:
            train_df[col] = train_df.groupby('site_id')[col].ffill().bfill()

    coverage = train_df['air_temperature'].notna().mean()
    print(f"Weather coverage: {coverage:.2%}")

    gc.collect()
    return train_df


def feature_engineering(df):
    """
    Create features for model training
    """
    print("\n=== FEATURE ENGINEERING ===")

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Building features
    df['square_feet_log'] = np.log1p(df['square_feet'].fillna(df['square_feet'].median()))
    df['year_built'] = df['year_built'].fillna(df['year_built'].median())
    df['building_age'] = 2016 - df['year_built']

    # Weather features
    df['temp_diff'] = df['air_temperature'] - df['dew_temperature']

    print(f"Features created: {df.shape[1]} total columns")

    return df


def create_lag_features(df):
    """
    Create lag and rolling features (per building)
    """
    print("\n=== CREATING LAG FEATURES ===")

    df = df.sort_values(['building_id', 'timestamp'])

    # Lag features
    df['meter_reading_lag_24h'] = df.groupby('building_id')['meter_reading'].shift(24)

    # Rolling statistics
    df['meter_reading_roll_mean_24h'] = df.groupby('building_id')['meter_reading'].transform(
        lambda x: x.rolling(24, min_periods=1).mean()
    )
    df['meter_reading_roll_std_24h'] = df.groupby('building_id')['meter_reading'].transform(
        lambda x: x.rolling(24, min_periods=1).std()
    )

    # Fill NaN values
    df['meter_reading_lag_24h'] = df['meter_reading_lag_24h'].fillna(
        df.groupby('building_id')['meter_reading'].transform('mean')
    )
    df['meter_reading_roll_std_24h'] = df['meter_reading_roll_std_24h'].fillna(0)

    print("Lag features created")

    return df


def time_based_split(df):
    """
    Split data by time (NO random shuffle!)
    Memory-optimized: uses boolean masks instead of copy
    """
    print("\n=== TIME-BASED SPLIT ===")

    # Convert string dates to datetime for comparison
    train_end_dt = pd.Timestamp(TRAIN_END)
    val_end_dt = pd.Timestamp(VAL_END)

    # Create masks
    train_mask = df['timestamp'] <= train_end_dt
    val_mask = (df['timestamp'] > train_end_dt) & (df['timestamp'] <= val_end_dt)
    test_mask = df['timestamp'] > val_end_dt

    # Split using loc (more memory efficient)
    train_set = df.loc[train_mask].reset_index(drop=True)
    val_set = df.loc[val_mask].reset_index(drop=True)
    test_set = df.loc[test_mask].reset_index(drop=True)

    # Free memory from original df
    del df
    gc.collect()

    print(f"Train: {len(train_set):,} rows ({train_set['timestamp'].min()} to {train_set['timestamp'].max()})")
    print(f"Val:   {len(val_set):,} rows ({val_set['timestamp'].min()} to {val_set['timestamp'].max()})")
    print(f"Test:  {len(test_set):,} rows ({test_set['timestamp'].min()} to {test_set['timestamp'].max()})")

    # Verify no leakage
    train_buildings = set(train_set['building_id'].unique())
    val_buildings = set(val_set['building_id'].unique())
    test_buildings = set(test_set['building_id'].unique())

    print(f"\nBuildings in all sets: {len(train_buildings & val_buildings & test_buildings)}")

    return train_set, val_set, test_set


def create_sequences(df, seq_length=168, target_horizon=24, stride=SEQUENCE_STRIDE, max_buildings=MAX_BUILDINGS):
    """
    Create sequences for LSTM training
    Memory-optimized with stride and building limit
    """
    print(f"\nCreating sequences (seq_length={seq_length}, target_horizon={target_horizon}, stride={stride})...")

    feature_cols = [
        'meter_reading', 'air_temperature', 'dew_temperature',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'square_feet_log',
        'building_age', 'temp_diff', 'meter_reading_lag_24h',
        'meter_reading_roll_mean_24h'
    ]

    sequences = []
    targets = []
    building_ids = []
    timestamps = []

    # Limit buildings if needed
    unique_buildings = df['building_id'].unique()
    if max_buildings is not None and len(unique_buildings) > max_buildings:
        print(f"⚠️ Limiting to {max_buildings} buildings (from {len(unique_buildings)})")
        np.random.seed(42)
        unique_buildings = np.random.choice(unique_buildings, max_buildings, replace=False)

    for building_id in tqdm(unique_buildings, desc="Processing buildings"):
        building_data = df[df['building_id'] == building_id].sort_values('timestamp')

        if len(building_data) < seq_length + target_horizon:
            continue

        values = building_data[feature_cols].values.astype(np.float32)
        target_values = building_data['meter_reading'].values.astype(np.float32)
        ts_values = building_data['timestamp'].values

        # Create sliding windows with stride
        for i in range(0, len(values) - seq_length - target_horizon + 1, stride):
            seq = values[i:i+seq_length]
            target = target_values[i+seq_length:i+seq_length+target_horizon]

            sequences.append(seq)
            targets.append(target)
            building_ids.append(building_id)
            timestamps.append(ts_values[i+seq_length])

    print(f"Total sequences created: {len(sequences):,}")
    
    # Estimate memory
    est_memory_gb = (len(sequences) * seq_length * len(feature_cols) * 4) / (1024**3)
    print(f"Estimated memory: {est_memory_gb:.2f} GB")

    return {
        'sequences': np.array(sequences, dtype=np.float32),
        'targets': np.array(targets, dtype=np.float32),
        'building_ids': np.array(building_ids),
        'timestamps': np.array(timestamps)
    }


def prepare_building_embedding(train_df, val_df, test_df):
    """
    Prepare building ID embeddings
    """
    print("\n=== PREPARING BUILDING EMBEDDING ===")

    # Combine all building IDs
    all_buildings = np.concatenate([
        train_df['building_ids'],
        val_df['building_ids'],
        test_df['building_ids']
    ])

    # Fit encoder
    encoder = LabelEncoder()
    encoder.fit(all_buildings)

    # Transform
    train_encoded = encoder.transform(train_df['building_ids'])
    val_encoded = encoder.transform(val_df['building_ids'])
    test_encoded = encoder.transform(test_df['building_ids'])

    n_buildings = len(encoder.classes_)
    embedding_dim = min(50, (n_buildings + 1) // 2)

    print(f"Unique buildings: {n_buildings}")
    print(f"Embedding dimension: {embedding_dim}")

    return {
        'encoder': encoder,
        'train': train_encoded,
        'val': val_encoded,
        'test': test_encoded,
        'n_buildings': n_buildings,
        'embedding_dim': embedding_dim
    }


class ASHRAEDataset(Dataset):
    """
    PyTorch Dataset for ASHRAE
    """
    def __init__(self, sequences, targets, building_ids):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.building_ids = torch.LongTensor(building_ids)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'building_id': self.building_ids[idx],
            'target': self.targets[idx]
        }


def create_dataloaders(train_data, val_data, test_data, batch_size=64):
    """
    Create PyTorch DataLoaders
    """
    print("\n=== CREATING DATALOADERS ===")

    train_dataset = ASHRAEDataset(
        train_data['sequences'],
        train_data['targets'],
        train_data['building_ids']
    )
    val_dataset = ASHRAEDataset(
        val_data['sequences'],
        val_data['targets'],
        val_data['building_ids']
    )
    test_dataset = ASHRAEDataset(
        test_data['sequences'],
        test_data['targets'],
        test_data['building_ids']
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def run_preprocessing(data_path='data', sample_size=SAMPLE_SIZE):
    """
    Run full preprocessing pipeline
    Memory-optimized version
    """
    print("=" * 60)
    print("🔧 PREPROCESSING PIPELINE")
    print("=" * 60)

    # Check available memory (optional - skip if psutil not available)
    try:
        import psutil
        available_ram = psutil.virtual_memory().available / (1024**3)
        print(f"Available RAM: {available_ram:.1f} GB")
    except ImportError:
        print("(psutil not available - skipping memory check)")

    # Load and clean (building sampling happens here now)
    train_df, weather_df, building_meta = load_and_clean_data(data_path, sample_size)

    # Merge
    train_full = merge_data(train_df, weather_df, building_meta)
    del train_df, weather_df
    gc.collect()

    # Feature engineering
    train_full = feature_engineering(train_full)
    train_full = create_lag_features(train_full)

    # Convert to float32 for memory efficiency
    float_cols = train_full.select_dtypes(include=['float64']).columns
    train_full[float_cols] = train_full[float_cols].astype('float32')
    gc.collect()

    print(f"\n📊 Memory usage: {train_full.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Split
    train_set, val_set, test_set = time_based_split(train_full)

    # Check if splits are valid
    if len(val_set) < 1000:
        print(f"\n⚠️ WARNING: Validation set too small ({len(val_set)} rows)")
        print("This may indicate data doesn't span the full time range.")
    if len(test_set) < 1000:
        print(f"⚠️ WARNING: Test set too small ({len(test_set)} rows)")

    # Create sequences with progress
    print("\n📦 Creating sequences...")
    train_seq = create_sequences(train_set)
    del train_set
    gc.collect()

    val_seq = create_sequences(val_set)
    del val_set
    gc.collect()

    test_seq = create_sequences(test_set)
    del test_set
    gc.collect()

    print(f"\nTrain sequences: {train_seq['sequences'].shape}")
    print(f"Val sequences: {val_seq['sequences'].shape if len(val_seq['sequences']) > 0 else '(empty)'}")
    print(f"Test sequences: {test_seq['sequences'].shape if len(test_seq['sequences']) > 0 else '(empty)'}")

    # Validate we have data
    if len(val_seq['sequences']) == 0 or len(test_seq['sequences']) == 0:
        print("\n❌ ERROR: Val or Test sequences are empty!")
        print("Check that your data spans 2016-2017.")
        raise ValueError("Validation or test set is empty after sequence creation")

    # Building embedding
    embedding_info = prepare_building_embedding(train_seq, val_seq, test_seq)

    # Update sequences with encoded building IDs
    train_seq['building_ids'] = embedding_info['train']
    val_seq['building_ids'] = embedding_info['val']
    test_seq['building_ids'] = embedding_info['test']

    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 60)

    return {
        'train': train_seq,
        'val': val_seq,
        'test': test_seq,
        'embedding_info': embedding_info,
        'building_meta': building_meta
    }


if __name__ == "__main__":
    import joblib
    from paths import PROCESSED_DIR, ensure_artifact_dirs

    ensure_artifact_dirs()

    # Run preprocessing
    data = run_preprocessing()

    # Save processed data
    print("\nSaving processed data...")
    np.save(PROCESSED_DIR / 'processed_train.npy', data['train'])
    np.save(PROCESSED_DIR / 'processed_val.npy', data['val'])
    np.save(PROCESSED_DIR / 'processed_test.npy', data['test'])
    joblib.dump(data['embedding_info'], PROCESSED_DIR / 'embedding_info.pkl')
    print("✅ Data saved!")
