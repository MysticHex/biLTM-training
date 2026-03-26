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
        'sequences': np.load('processed_train_seq.npy'),
        'targets': np.load('processed_train_tgt.npy'),
        'building_ids': np.load('processed_train_bid.npy')
    }
    val_data = {
        'sequences': np.load('processed_val_seq.npy'),
        'targets': np.load('processed_val_tgt.npy'),
        'building_ids': np.load('processed_val_bid.npy')
    }
    test_data = {
        'sequences': np.load('processed_test_seq.npy'),
        'targets': np.load('processed_test_tgt.npy'),
        'building_ids': np.load('processed_test_bid.npy')
    }
    embedding_info = joblib.load('embedding_info.pkl')

    print(f"Train sequences: {train_data['sequences'].shape}")
    print(f"Val sequences: {val_data['sequences'].shape}")
    print(f"Test sequences: {test_data['sequences'].shape}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=64
    )

    # Load best params atau gunakan default
    if Path('best_params.json').exists():
        with open('best_params.json', 'r') as f:
            best_params = json.load(f)
        print("\n✅ Loaded best hyperparameters from best_params.json")
    else:
        print("\n⚠️ best_params.json not found, using defaults")
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
        plt.savefig('training_history.png', dpi=150)
        plt.show()

        # Final evaluation on test set
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 80)

        # Load best model
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        test_results = get_predictions(model, test_loader, device)
        test_metrics = calculate_metrics(test_results['preds'], test_results['targets'])

        print("\n📊 TEST RESULTS:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save metrics
        with open('test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)

        print("\n✅ Training complete!")
        print("\nFiles saved:")
        print("  - best_model.pth (model checkpoint)")
        print("  - training_history.png")
        print("  - test_metrics.json")

        print("\nNext steps:")
        print("  1. Run XAI analysis dengan xai.py")
        print("  2. Generate dashboard dengan dashboard.py")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted.")
        print("Best model saved so far: best_model.pth")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
