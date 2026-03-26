"""
run_xai_dashboard.py - Generate XAI Visualizations & Dashboard

Script untuk menjalankan XAI analysis dan generate retrofit dashboard.

Usage:
    python run_xai_dashboard.py
"""

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

# Import modules
from model import create_model
from preprocessing import ASHRAEDataset
from evaluate import get_predictions, calculate_metrics, create_anomaly_features, train_anomaly_classifier
from xai import visualize_multihead_attention, visualize_attention_timeline
from dashboard import create_retrofit_dashboard, print_summary_table


def main():
    print("=" * 80)
    print("🎨 ATTNRETROFIT - XAI & DASHBOARD GENERATION")
    print("=" * 80)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📟 Device: {device}")

    # Load processed data
    print("\n" + "=" * 80)
    print("LOADING DATA & MODEL")
    print("=" * 80)

    train_data = {
        'sequences': np.load('processed_train_seq.npy'),
        'targets': np.load('processed_train_tgt.npy'),
        'building_ids': np.load('processed_train_bid.npy')
    }
    test_data = {
        'sequences': np.load('processed_test_seq.npy'),
        'targets': np.load('processed_test_tgt.npy'),
        'building_ids': np.load('processed_test_bid.npy')
    }
    embedding_info = joblib.load('embedding_info.pkl')

    print(f"Train sequences: {train_data['sequences'].shape}")
    print(f"Test sequences: {test_data['sequences'].shape}")

    # Load best params
    if Path('best_params.json').exists():
        with open('best_params.json', 'r') as f:
            best_params = json.load(f)
    else:
        best_params = {
            'hidden_dim': 128,
            'num_layers': 2,
            'num_attention_heads': 2,
            'dropout': 0.3
        }

    # Create model config
    config = {
        'n_buildings': embedding_info['n_buildings'],
        'embedding_dim': embedding_info['embedding_dim'],
        'input_dim': train_data['sequences'].shape[2],
        'hidden_dim': best_params.get('hidden_dim', 128),
        'num_layers': best_params.get('num_layers', 2),
        'num_attention_heads': best_params.get('num_attention_heads', 2),
        'output_horizon': 24,
        'dropout': best_params.get('dropout', 0.3)
    }

    # Create and load model
    model = create_model(config, device)
    
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded from best_model.pth")

    # Create test dataloader
    test_dataset = ASHRAEDataset(
        test_data['sequences'],
        test_data['targets'],
        test_data['building_ids']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    # Get predictions
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    results = get_predictions(model, test_loader, device)
    metrics = calculate_metrics(results['preds'], results['targets'])

    print("\n📊 TEST METRICS:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # XAI: Attention Visualization
    print("\n" + "=" * 80)
    print("XAI: ATTENTION VISUALIZATION")
    print("=" * 80)

    # Get a sample for visualization
    sample_idx = 0
    sample_seq = torch.FloatTensor(test_data['sequences'][sample_idx:sample_idx+1]).to(device)
    sample_bid = torch.LongTensor(test_data['building_ids'][sample_idx:sample_idx+1]).to(device)

    print("\n🔍 Generating multi-head attention visualization...")
    try:
        visualize_multihead_attention(model, sample_seq, sample_bid, device)
        print("✅ Saved: attention_multihead.png")
    except Exception as e:
        print(f"⚠️ Could not generate attention visualization: {e}")

    print("\n🔍 Generating attention timeline...")
    try:
        visualize_attention_timeline(model, sample_seq, sample_bid, device)
        print("✅ Saved: attention_timeline.png")
    except Exception as e:
        print(f"⚠️ Could not generate timeline: {e}")

    # Anomaly Detection
    print("\n" + "=" * 80)
    print("ANOMALY DETECTION")
    print("=" * 80)

    # Calculate residuals
    residuals = torch.abs(results['preds'] - results['targets'])
    residual_mean = residuals.mean(dim=1)

    print(f"Residual stats: mean={residual_mean.mean():.2f}, std={residual_mean.std():.2f}")
    print(f"Max residual: {residual_mean.max():.2f}")

    # Create anomaly features
    features = create_anomaly_features(results, residual_mean)

    # Train anomaly classifier (using residual threshold)
    threshold_percentile = 95
    threshold = residual_mean.quantile(threshold_percentile / 100).item()
    is_anomaly = (residual_mean > threshold).numpy()

    print(f"\nAnomaly threshold (P{threshold_percentile}): {threshold:.2f}")
    print(f"Anomalies detected: {is_anomaly.sum()} / {len(is_anomaly)} ({is_anomaly.mean()*100:.1f}%)")

    # Create anomaly DataFrame
    anomaly_df = pd.DataFrame({
        'building_id': results['buildings'].numpy(),
        'residual_mean': residual_mean.numpy(),
        'pred_mean': results['preds'].mean(dim=1).numpy(),
        'target_mean': results['targets'].mean(dim=1).numpy(),
        'is_anomaly': is_anomaly
    })

    # Calculate combined score (simplified without SHAP)
    anomaly_df['combined_score'] = (
        (residual_mean - residual_mean.mean()) / residual_mean.std()
    ).numpy()

    # Sort by anomaly score
    anomaly_df = anomaly_df.sort_values('combined_score', ascending=False)

    # Load building metadata
    building_meta = pd.read_csv('data/building_metadata.csv')

    # Dashboard
    print("\n" + "=" * 80)
    print("GENERATING DASHBOARD")
    print("=" * 80)

    try:
        report_df = create_retrofit_dashboard(anomaly_df, building_meta, results, top_n=20)
        print_summary_table(report_df, metrics)
    except Exception as e:
        print(f"⚠️ Dashboard error: {e}")
        # Fallback: simple report
        print("\n📋 TOP 10 ANOMALOUS BUILDINGS:")
        print(anomaly_df.head(10).to_string())

    # Save anomaly report
    anomaly_df.to_csv('anomaly_report.csv', index=False)
    print("\n✅ Saved: anomaly_report.csv")

    # Summary
    print("\n" + "=" * 80)
    print("✅ XAI & DASHBOARD COMPLETE")
    print("=" * 80)
    print("\nFiles generated:")
    print("  - attention_multihead.png")
    print("  - attention_timeline.png")
    print("  - retrofit_dashboard.png")
    print("  - retrofit_priority_report.csv")
    print("  - anomaly_report.csv")

    print("\n🎉 Project AttnRetrofit selesai!")


if __name__ == "__main__":
    main()
