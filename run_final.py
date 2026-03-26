"""
run_final.py - Final Model Training dengan Best Hyperparameters

Script untuk melatih model final dengan hyperparameters terbaik dari Optuna.

Usage:
    python run_final.py
"""

import os
import sys
import torch
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from paths import PROCESSED_DIR, CONFIG_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR, ensure_artifact_dirs

# Set seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Import modules
from preprocessing import create_dataloaders
from model import create_model
from train import train_model, validate
from evaluate import get_predictions, calculate_metrics
from xai import calculate_shap_values, visualize_global_shap, visualize_local_shap
from dashboard import create_retrofit_dashboard

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
    best_model_path = MODELS_DIR / 'best_model.pth'
    metrics_path = METRICS_DIR / 'test_metrics.json'
    history_plot_path = PLOTS_DIR / 'training_history.png'

    print("=" * 80)
    print("🚀 ATTNRETROFIT - FINAL MODEL TRAINING")
    print("=" * 80)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📟 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load processed data
    print("\n" + "=" * 80)
    print("LOADING PROCESSED DATA")
    print("=" * 80)

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

    print(f"Train sequences: {train_data['sequences'].shape}")
    print(f"Val sequences: {val_data['sequences'].shape}")
    print(f"Test sequences: {test_data['sequences'].shape}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=64
    )

    # Load best params atau gunakan default
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        print(f"\n✅ Loaded best hyperparameters from {best_params_path}")
    else:
        print(f"\n⚠️ {best_params_path} not found, using defaults")
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

    # Final config
    final_config = {
        'n_buildings': embedding_info['n_buildings'],
        'embedding_dim': embedding_info['embedding_dim'],
        'input_dim': train_data['sequences'].shape[2],
        'hidden_dim': best_params.get('hidden_dim', 128),
        'num_layers': best_params.get('num_layers', 2),
        'num_attention_heads': best_params.get('num_attention_heads', 4),
        'output_horizon': 24,
        'dropout': best_params.get('dropout', 0.3),
        'lr': best_params.get('lr', 5e-4),
        'weight_decay': best_params.get('weight_decay', 1e-5),
        'loss_alpha': best_params.get('loss_alpha', 0.5),
        'epochs': 100,
        'patience': 10
    }

    print("\n" + "=" * 80)
    print("FINAL MODEL CONFIGURATION")
    print("=" * 80)
    for key, value in final_config.items():
        if key != 'epochs' and key != 'patience':
            print(f"  {key}: {value}")

    # Create model
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    print("This will train for up to 100 epochs dengan early stopping.")
    print()

    model = create_model(final_config, device)

    try:
        # Train
        history = train_model(
            model,
            train_loader,
            val_loader,
            final_config,
            device
        )

        # Plot training history
        print("\n📊 Plotting training history...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0,0].plot(history['train_loss'], label='Train')
        axes[0,0].plot(history['val_loss'], label='Val')
        axes[0,0].set_title('Loss')
        axes[0,0].legend()

        axes[0,1].plot(history['val_rmse'])
        axes[0,1].set_title('Validation RMSE')

        axes[1,0].plot(history['val_mae'])
        axes[1,0].set_title('Validation MAE')

        axes[1,1].plot(history['val_rmsle'])
        axes[1,1].set_title('Validation RMSLE')

        plt.tight_layout()
        plt.savefig(history_plot_path, dpi=150)
        plt.show()

        # Final evaluation on test set
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 80)

        # Load best model
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_results = get_predictions(model, test_loader, device)
        test_metrics = calculate_metrics(test_results['preds'], test_results['targets'])

        print("\n📊 TEST RESULTS:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)

        print("\n✅ Training complete!")
        print("\nFiles saved:")
        print(f"  - {best_model_path} (model checkpoint)")
        print(f"  - {history_plot_path}")
        print(f"  - {metrics_path}")

        print("\nNext steps:")
        print("  1. Run XAI analysis dengan xai.py")
        print("  2. Generate dashboard dengan dashboard.py")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted.")
        print(f"Best model saved so far: {best_model_path}")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
