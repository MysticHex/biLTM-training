"""
 xai.py - Explainable AI (SHAP & Attention Visualization)

 Phase 7: XAI Integration
 - SHAP DeepExplainer (global & local)
 - Multi-head attention visualization
 - Per-head pattern analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from paths import PLOTS_DIR, ensure_artifact_dirs


class ModelWrapper:
    """Wrapper untuk SHAP compatibility"""
    def __init__(self, model, building_ids, device):
        self.model = model
        self.building_ids = building_ids
        self.device = device

    def __call__(self, sequences):
        if isinstance(sequences, np.ndarray):
            sequences = torch.FloatTensor(sequences).to(self.device)

        self.model.eval()
        with torch.no_grad():
            batch_size = sequences.shape[0]
            building_ids = self.building_ids[:batch_size].to(self.device)
            preds, _ = self.model(sequences, building_ids)
        return preds.cpu().numpy()


def calculate_shap_values(model, background_data, test_data, test_buildings,
                           feature_names, device='cuda', sample_size=50):
    """
    Calculate SHAP values untuk explainability
    """
    print("=== CALCULATING SHAP VALUES ===")

    # Sample untuk speed
    test_indices = np.random.choice(len(test_data), min(sample_size, len(test_data)), replace=False)
    test_sample = test_data[test_indices]
    test_bids = test_buildings[test_indices]

    # Wrapper
    model_wrapper = ModelWrapper(model, test_bids, device)

    # DeepExplainer
    print("Initializing DeepExplainer...")
    explainer = shap.DeepExplainer(model, background_data[:10])

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(test_sample)

    print(f"SHAP values shape: {np.array(shap_values).shape}")

    return shap_values, test_sample, test_indices


def visualize_global_shap(shap_values, test_data, feature_names):
    """
    Global feature importance (summary plot)
    """
    print("=== GLOBAL SHAP VISUALIZATION ===")
    ensure_artifact_dirs()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Aggregate SHAP values
    shap_array = np.array(shap_values)
    shap_importance = np.abs(shap_array).mean(axis=(0, 2))

    shap.summary_plot(
        shap_importance,
        test_data.mean(axis=1),
        feature_names=feature_names,
        show=False
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    output_path = PLOTS_DIR / 'shap_global.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Print top features
    shap_importance_mean = shap_importance.mean(axis=0)
    top_indices = np.argsort(shap_importance_mean)[::-1][:5]
    print("\n🏆 TOP 5 GLOBAL FEATURES:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {feature_names[idx]}: {shap_importance_mean[idx]:.4f}")


def visualize_local_shap(shap_values, test_data, test_indices, residual_errors,
                        feature_names, top_k=3):
    """
    Local SHAP untuk anomalous samples
    """
    print("=== LOCAL SHAP VISUALIZATION ===")
    ensure_artifact_dirs()

    # Find anomalous samples
    anomalous_indices = np.argsort(residual_errors[test_indices])[-top_k:]

    fig, axes = plt.subplots(1, top_k, figsize=(18, 5))
    if top_k == 1:
        axes = [axes]

    shap_array = np.array(shap_values)

    for i, idx in enumerate(anomalous_indices):
        shap_local = shap_array[:, idx, :, :].mean(axis=0)
        shap_local_mean = np.abs(shap_local).mean(axis=0)

        colors = ['red' if v > np.percentile(shap_local_mean, 75) else 'steelblue'
                  for v in shap_local_mean]

        axes[i].barh(feature_names, shap_local_mean, color=colors)
        axes[i].set_xlabel('Mean |SHAP Value|')
        axes[i].set_title(f'Anomaly #{idx}\nResidual: {residual_errors[test_indices][idx]:.2f}')
        axes[i].invert_yaxis()

    plt.tight_layout()
    output_path = PLOTS_DIR / 'shap_local.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_multihead_attention(model, sample_sequence, sample_building_id,
                                  device='cuda'):
    """
    Visualisasi multi-head attention weights
    """
    print("=== MULTI-HEAD ATTENTION VISUALIZATION ===")
    ensure_artifact_dirs()

    model.eval()
    with torch.no_grad():
        _, attn_weights = model(sample_sequence, sample_building_id)

    attn_weights = attn_weights.cpu().numpy()[0]
    num_heads = attn_weights.shape[0]

    fig, axes = plt.subplots(1, num_heads, figsize=(20, 4))
    if num_heads == 1:
        axes = [axes]

    for i in range(num_heads):
        im = axes[i].imshow(attn_weights[i], cmap='viridis', aspect='auto')
        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key Position (Time)')
        axes[i].set_ylabel('Query Position (Time)')
        plt.colorbar(im, ax=axes[i])

    plt.suptitle('Multi-Head Attention Weights', fontsize=14)
    plt.tight_layout()
    output_path = PLOTS_DIR / 'attention_multihead.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Analyze head patterns
    print("\n🔍 HEAD ANALYSIS:")
    for i in range(num_heads):
        head_weights = attn_weights[i]
        avg_attention = head_weights.mean(axis=0)
        peak_idx = np.argmax(avg_attention)

        recent_focus = avg_attention[-24:].mean()
        distant_focus = avg_attention[:-24].mean() if len(avg_attention) > 24 else 0

        if recent_focus > distant_focus * 1.5:
            pattern = "Recent focus (last 24h)"
        elif distant_focus > recent_focus * 1.5:
            pattern = "Distant past focus"
        else:
            pattern = "Balanced attention"

        print(f"  Head {i+1}: {pattern} | Peak at t-{len(avg_attention)-peak_idx}h")


def visualize_attention_timeline(model, sample_sequence, sample_building_id,
                                 device='cuda'):
    """
    Timeline attention untuk satu sample
    """
    ensure_artifact_dirs()
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(sample_sequence, sample_building_id)

    attn_weights = attn_weights.cpu().numpy()[0]

    # Average across heads
    avg_attention = attn_weights.mean(axis=0).mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(avg_attention, marker='o', linewidth=2, markersize=4, color='steelblue')
    ax.axhline(y=avg_attention.mean(), color='r', linestyle='--', label='Mean')
    ax.set_xlabel('Time Step (Hours ago)')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Temporal Attention Weights (Averaged)')
    ax.legend()

    plt.tight_layout()
    output_path = PLOTS_DIR / 'attention_timeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("XAI module - import and use functions")
