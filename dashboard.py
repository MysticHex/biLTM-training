"""
 dashboard.py - Retrofit Priority Dashboard

 Phase 8: Dashboard Generation
 - Priority report CSV
 - 4-panel visualization
 - Summary metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_retrofit_dashboard(anomaly_df, building_meta, results, top_n=20):
    """
    Create comprehensive dashboard
    """
    print("=" * 60)
    print("🏢 RETROFIT PRIORITY DASHBOARD")
    print("=" * 60)

    # Generate report
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
            'residual_error': row.get('residual_mean', 0),
            'avg_consumption_kwh': avg_consumption,
            'potential_savings_kwh': avg_consumption * 0.2,
            'priority': 'HIGH' if row['combined_score'] > anomaly_df['combined_score'].quantile(0.95) else 'MEDIUM'
        })

    report_df = pd.DataFrame(report)

    # Save CSV
    report_df.to_csv('retrofit_priority_report.csv', index=False)
    print(f"\n✅ Report saved: retrofit_priority_report.csv")

    # Create dashboard visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Priority by Building Type
    type_priority = report_df.groupby('primary_use')['anomaly_score'].mean().sort_values(ascending=False)
    type_priority.plot(kind='bar', ax=axes[0,0], color='steelblue')
    axes[0,0].set_title('Average Anomaly Score by Building Type')
    axes[0,0].set_ylabel('Anomaly Score')
    axes[0,0].tick_params(axis='x', rotation=45)

    # 2. Potential Savings
    axes[0,1].scatter(report_df['square_feet'], report_df['potential_savings_kwh'],
                      c=report_df['anomaly_score'], cmap='Reds', s=100)
    axes[0,1].set_xlabel('Square Feet')
    axes[0,1].set_ylabel('Potential Savings (kWh)')
    axes[0,1].set_title('Potential Savings vs Building Size')

    # 3. Top Anomalies
    axes[1,0].barh(report_df['building_id'].astype(str)[:10], report_df['anomaly_score'][:10], color='coral')
    axes[1,0].set_xlabel('Anomaly Score')
    axes[1,0].set_title('Top 10 Buildings by Anomaly Score')
    axes[1,0].invert_yaxis()

    # 4. Summary metrics
    summary_text = f"""
    RETROFIT SUMMARY
    ================
    Total Buildings Analyzed: {len(anomaly_df)}
    High Priority: {(anomaly_df['combined_score'] > anomaly_df['combined_score'].quantile(0.95)).sum()}
    Medium Priority: {((anomaly_df['combined_score'] > anomaly_df['combined_score'].quantile(0.90)) &
                    (anomaly_df['combined_score'] <= anomaly_df['combined_score'].quantile(0.95))).sum()}

    Average Anomaly Score: {anomaly_df['combined_score'].mean():.4f}
    Max Anomaly Score: {anomaly_df['combined_score'].max():.4f}

    Total Potential Savings: {report_df['potential_savings_kwh'].sum():,.0f} kWh
    """
    axes[1,1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center', transform=axes[1,1].transAxes)
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.savefig('retrofit_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Dashboard saved: retrofit_dashboard.png")

    return report_df


def print_summary_table(report_df, metrics=None):
    """
    Print summary table
    """
    print("\n" + "=" * 80)
    print("🏆 TOP 10 RETROFIT PRIORITIES")
    print("=" * 80)
    print(report_df.head(10).to_string(index=False))

    if metrics:
        print("\n" + "=" * 80)
        print("📊 MODEL PERFORMANCE")
        print("=" * 80)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    print("Dashboard module - import and use create_retrofit_dashboard()")
