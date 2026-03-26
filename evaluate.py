"""
 evaluate.py - Evaluation & Anomaly Detection

 Phase 6: Evaluation & Anomaly Detection
 - Model evaluation (RMSE, MAE, RMSLE)
 - Residual error calculation
 - Learned threshold model (Random Forest)
 - Combined anomaly scoring
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib


def get_predictions(model, loader, device='cuda'):
    """Get predictions untuk seluruh dataset"""
    model.eval()
    all_preds = []
    all_targets = []
    all_buildings = []
    all_sequences = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions"):
            sequence = batch['sequence'].to(device)
            building_id = batch['building_id'].to(device)
            target = batch['target']

            pred, _ = model(sequence, building_id)

            all_preds.append(pred.cpu())
            all_targets.append(target)
            all_buildings.append(building_id.cpu())
            all_sequences.append(sequence.cpu())

    return {
        'preds': torch.cat(all_preds, dim=0),
        'targets': torch.cat(all_targets, dim=0),
        'buildings': torch.cat(all_buildings, dim=0),
        'sequences': torch.cat(all_sequences, dim=0)
    }


def calculate_metrics(preds, targets):
    """Calculate evaluation metrics"""
    rmse = torch.sqrt(F.mse_loss(preds, targets)).item()
    mae = F.l1_loss(preds, targets).item()
    rmsle = torch.sqrt(F.mse_loss(
        torch.log1p(preds), torch.log1p(targets)
    )).item()

    return {'RMSE': rmse, 'MAE': mae, 'RMSLE': rmsle}


def create_anomaly_features(results, residual_mean, shap_values=None, building_meta=None):
    """
    Create features untuk anomaly detection
    """
    n_samples = len(residual_mean)
    features = pd.DataFrame()

    # Residual-based features
    residuals = torch.abs(results['preds'] - results['targets'])
    features['residual_mean'] = residual_mean.numpy()
    features['residual_std'] = residuals.std(dim=1).numpy()
    features['residual_max'] = residuals.max(dim=1)[0].numpy()

    # Prediction vs target
    features['pred_mean'] = results['preds'].mean(dim=1).numpy()
    features['target_mean'] = results['targets'].mean(dim=1).numpy()
    features['pred_target_ratio'] = features['pred_mean'] / (features['target_mean'] + 1e-6)

    # Building characteristics
    if building_meta is not None:
        for bid in results['buildings']:
            bid_int = bid.item()
            row = building_meta[building_meta['building_id'] == bid_int]
            if len(row) > 0:
                features.loc[len(features), 'square_feet'] = row['square_feet'].iloc[0]
                features.loc[len(features)-1, 'building_age'] = 2016 - row['year_built'].iloc[0]

    # SHAP-based
    if shap_values is not None:
        shap_importance = np.abs(shap_values).mean(axis=(0, 2, 3))
        features['shap_importance'] = shap_importance[:n_samples]
    else:
        features['shap_importance'] = 0

    return features.fillna(0)


def train_anomaly_classifier(train_features, train_residual_mean, percentile=95):
    """
    Train Random Forest classifier untuk anomaly detection
    """
    print("Training anomaly classifier...")

    threshold = train_residual_mean.quantile(percentile / 100)
    labels = (train_residual_mean > threshold).long().numpy()

    print(f"Anomaly threshold (P{percentile}): {threshold:.4f}")
    print(f"Anomalies: {labels.sum()} / {len(labels)} ({labels.mean()*100:.2f}%)")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )

    clf.fit(train_features, labels)

    return clf, threshold


def calculate_combined_anomaly_score(results, residual_mean, features, clf):
    """
    Combined scoring: Residual + ML + SHAP
    """
    # Normalize residual
    residual_score = (residual_mean - residual_mean.mean()) / residual_mean.std()

    # ML probability
    anomaly_prob = clf.predict_proba(features)[:, 1]

    # Combined score (40% residual + 40% ML + 20% placeholder untuk SHAP)
    combined = 0.4 * residual_score.numpy() + 0.4 * anomaly_prob + 0.2 * np.random.random(len(residual_mean))

    return combined


def generate_anomaly_report(results, anomaly_df, building_meta, top_n=10):
    """
    Generate retrofit priority report
    """
    report = []

    for idx, row in anomaly_df.head(top_n).iterrows():
        building_id = int(row['building_id'])
        building_info = building_meta[building_meta['building_id'] == building_id]

        if len(building_info) == 0:
            continue

        building_info = building_info.iloc[0]
        avg_consumption = results['targets'][idx].mean().item()

        report.append({
            'rank': len(report) + 1,
            'building_id': building_id,
            'primary_use': building_info.get('primary_use', 'Unknown'),
            'square_feet': building_info.get('square_feet', 0),
            'anomaly_score': row['combined_score'],
            'avg_consumption_kwh': avg_consumption,
            'potential_savings_kwh': avg_consumption * 0.2,
            'priority': 'HIGH' if row['combined_score'] > anomaly_df['combined_score'].quantile(0.95) else 'MEDIUM'
        })

    return pd.DataFrame(report)


if __name__ == "__main__":
    print("Evaluate module - import and use functions")
