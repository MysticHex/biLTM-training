"""
run_training.py - Training Pipeline dengan Hyperparameter Tuning

Script untuk menjalankan preprocessing, hyperparameter tuning dengan Optuna,
dan training model final.

Usage:
    python run_training.py
"""

import os
import sys
import torch
import numpy as np
import joblib
from pathlib import Path
import json
import gc
from paths import PROCESSED_DIR, CONFIG_DIR, ensure_artifact_dirs

# Set seeds untuk reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Import modules
from preprocessing import run_preprocessing, create_dataloaders
from model import create_model
from train import create_optuna_study, train_model, ASHRAELoss

def main():
    ensure_artifact_dirs()

    train_seq_path = PROCESSED_DIR / 'processed_train_seq.npy'
    train_tgt_path = PROCESSED_DIR / 'processed_train_tgt.npy'
    train_bid_path = PROCESSED_DIR / 'processed_train_bid.npy'
    val_seq_path = PROCESSED_DIR / 'processed_val_seq.npy'
    val_tgt_path = PROCESSED_DIR / 'processed_val_tgt.npy'
    val_bid_path = PROCESSED_DIR / 'processed_val_bid.npy'
    test_seq_path = PROCESSED_DIR / 'processed_test_seq.npy'
    test_tgt_path = PROCESSED_DIR / 'processed_test_tgt.npy'
    test_bid_path = PROCESSED_DIR / 'processed_test_bid.npy'
    embedding_info_path = PROCESSED_DIR / 'embedding_info.pkl'
    best_params_path = CONFIG_DIR / 'best_params.json'

    print("=" * 80)
    print("🚀 ATTNRETROFIT - TRAINING PIPELINE")
    print("=" * 80)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📟 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")

    # Step 1: Preprocessing
    print("\n" + "=" * 80)
    print("STEP 1: PREPROCESSING")
    print("=" * 80)

    if not train_seq_path.exists():
        print("Running preprocessing...")
        data = run_preprocessing(data_path='data')

        # Save processed data - memory efficient (separate arrays)
        print("\n💾 Saving processed data (memory-efficient)...")
        
        # Save train components separately
        np.save(train_seq_path, data['train']['sequences'].astype(np.float32))
        np.save(train_tgt_path, data['train']['targets'].astype(np.float32))
        np.save(train_bid_path, data['train']['building_ids'])
        gc.collect()
        
        # Save val components
        np.save(val_seq_path, data['val']['sequences'].astype(np.float32))
        np.save(val_tgt_path, data['val']['targets'].astype(np.float32))
        np.save(val_bid_path, data['val']['building_ids'])
        gc.collect()
        
        # Save test components
        np.save(test_seq_path, data['test']['sequences'].astype(np.float32))
        np.save(test_tgt_path, data['test']['targets'].astype(np.float32))
        np.save(test_bid_path, data['test']['building_ids'])
        gc.collect()
        
        joblib.dump(data['embedding_info'], embedding_info_path)
        print("✅ Data saved!")
    else:
        print("Loading existing processed data...")
        train_data = {
            'sequences': np.load(train_seq_path),
            'targets': np.load(train_tgt_path),
            'building_ids': np.load(train_bid_path)
        }
        val_data = {
            'sequences': np.load(val_seq_path),
            'targets': np.load(val_tgt_path),
            'building_ids': np.load(val_bid_path)
        }
        test_data = {
            'sequences': np.load(test_seq_path),
            'targets': np.load(test_tgt_path),
            'building_ids': np.load(test_bid_path)
        }
        embedding_info = joblib.load(embedding_info_path)

        data = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'embedding_info': embedding_info
        }
        print("✅ Data loaded!")

    # Step 2: Create DataLoaders
    print("\n" + "=" * 80)
    print("STEP 2: CREATING DATALOADERS")
    print("=" * 80)

    # Use smaller batch size for tuning
    train_loader, val_loader, test_loader = create_dataloaders(
        data['train'],
        data['val'],
        data['test'],
        batch_size=64
    )

    # Base config
    base_config = {
        'n_buildings': data['embedding_info']['n_buildings'],
        'embedding_dim': data['embedding_info']['embedding_dim'],
        'input_dim': data['train']['sequences'].shape[2],
        'output_horizon': 24
    }

    print(f"\nBase config:")
    print(f"  Buildings: {base_config['n_buildings']}")
    print(f"  Embedding dim: {base_config['embedding_dim']}")
    print(f"  Input dim: {base_config['input_dim']}")
    print(f"  Sequence shape: {data['train']['sequences'].shape}")

    # Step 3: Hyperparameter Tuning dengan Optuna
    print("\n" + "=" * 80)
    print("STEP 3: HYPERPARAMETER TUNING (Optuna)")
    print("=" * 80)

    print("\nStarting Optuna study...")
    print("This will take approximately 1-2 hours depending on your GPU.")
    print("Each trial trains for 5 epochs dengan early stopping.")
    print()

    try:
        best_params = create_optuna_study(
            train_loader,
            val_loader,
            base_config,
            device=device
        )

        print("\n" + "=" * 80)
        print("🏆 BEST HYPERPARAMETERS FOUND")
        print("=" * 80)
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Save best params
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n✅ Best params saved to {best_params_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user.")
        print("You can resume by loading the saved study.")
        return
    except Exception as e:
        print(f"\n❌ Error during hyperparameter tuning: {e}")
        print("Using default parameters...")
        best_params = {
            'hidden_dim': 128,
            'num_layers': 2,
            'num_attention_heads': 4,
            'dropout': 0.3,
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'batch_size': 64,
            'loss_alpha': 0.5
        }

    print("\n" + "=" * 80)
    print("✅ STEP 3 COMPLETE")
    print("=" * 80)
    print("\nNext: Run run_final.py untuk training final model")
    print("dengan best hyperparameters.")

if __name__ == "__main__":
    main()
