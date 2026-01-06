"""
Visualization for anomaly detection results.
Run this after main.py --demo to see charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data():
    """Load the generated data and results"""
    transactions = pd.read_csv('data/sample_payroll.csv')
    ground_truth = pd.read_csv('data/ground_truth.csv')

    # Merge to get labels
    data = transactions.merge(ground_truth[['transaction_id', 'is_anomaly', 'anomaly_type']],
                              on='transaction_id', how='left')
    return data


def plot_anomaly_distribution(data, save_path='reports/'):
    """Show distribution of anomaly scores"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Score distribution by actual label
    ax1 = axes[0, 0]

    # We need to compute scores - for now show amount distribution
    normal = data[data['is_anomaly'] == False]['total_amount']
    anomaly = data[data['is_anomaly'] == True]['total_amount']

    ax1.hist(normal, bins=50, alpha=0.7, label=f'Normal (n={len(normal)})', color='steelblue')
    ax1.hist(anomaly, bins=20, alpha=0.7, label=f'Anomaly (n={len(anomaly)})', color='crimson')
    ax1.set_xlabel('Total Amount ($)')
    ax1.set_ylabel('Count')
    ax1.set_title('Transaction Amount Distribution')
    ax1.legend()

    # 2. Anomalies by department
    ax2 = axes[0, 1]
    dept_anomalies = data[data['is_anomaly'] == True].groupby('department').size()
    dept_total = data.groupby('department').size()
    dept_rate = (dept_anomalies / dept_total * 100).fillna(0)

    colors = ['crimson' if x > 2 else 'steelblue' for x in dept_rate.values]
    bars = ax2.bar(dept_rate.index, dept_rate.values, color=colors)
    ax2.set_xlabel('Department')
    ax2.set_ylabel('Anomaly Rate (%)')
    ax2.set_title('Anomaly Rate by Department')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Expected (2%)')
    ax2.legend()

    # 3. Anomaly types breakdown
    ax3 = axes[1, 0]
    anomaly_types = data[data['is_anomaly'] == True]['anomaly_type'].value_counts()

    colors_pie = plt.cm.Set2(np.linspace(0, 1, len(anomaly_types)))
    wedges, texts, autotexts = ax3.pie(anomaly_types.values, labels=anomaly_types.index,
                                        autopct='%1.1f%%', colors=colors_pie)
    ax3.set_title('Types of Injected Anomalies')

    # 4. Overtime analysis
    ax4 = axes[1, 1]

    normal_ot = data[data['is_anomaly'] == False]['overtime_hours']
    anomaly_ot = data[data['is_anomaly'] == True]['overtime_hours']

    bp = ax4.boxplot([normal_ot, anomaly_ot], labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('crimson')
    ax4.set_ylabel('Overtime Hours')
    ax4.set_title('Overtime Hours: Normal vs Anomaly')

    plt.tight_layout()
    plt.savefig(f'{save_path}anomaly_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}anomaly_analysis.png")


def plot_time_series(data, save_path='reports/'):
    """Show anomalies over time"""
    fig, ax = plt.subplots(figsize=(14, 5))

    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    data['month'] = data['transaction_date'].dt.to_period('M')

    monthly = data.groupby('month').agg({
        'transaction_id': 'count',
        'is_anomaly': 'sum'
    }).rename(columns={'transaction_id': 'total', 'is_anomaly': 'anomalies'})

    monthly['anomaly_rate'] = monthly['anomalies'] / monthly['total'] * 100

    x = range(len(monthly))
    ax.bar(x, monthly['total'], alpha=0.3, label='Total Transactions', color='steelblue')
    ax2 = ax.twinx()
    ax2.plot(x, monthly['anomaly_rate'], 'ro-', label='Anomaly Rate %', linewidth=2, markersize=6)
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('Month')
    ax.set_ylabel('Transaction Count')
    ax2.set_ylabel('Anomaly Rate (%)', color='red')
    ax.set_title('Transactions and Anomaly Rate Over Time')
    ax.set_xticks(x[::3])
    ax.set_xticklabels([str(m) for m in monthly.index[::3]], rotation=45)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{save_path}time_series.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}time_series.png")


def plot_feature_analysis(data, save_path='reports/'):
    """Analyze key features for anomaly detection"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Salary vs Overtime scatter
    ax1 = axes[0]
    normal = data[data['is_anomaly'] == False]
    anomaly = data[data['is_anomaly'] == True]

    ax1.scatter(normal['base_amount'], normal['overtime_hours'],
                alpha=0.3, s=10, c='steelblue', label='Normal')
    ax1.scatter(anomaly['base_amount'], anomaly['overtime_hours'],
                alpha=0.8, s=50, c='crimson', marker='x', label='Anomaly')
    ax1.set_xlabel('Base Salary ($)')
    ax1.set_ylabel('Overtime Hours')
    ax1.set_title('Salary vs Overtime')
    ax1.legend()

    # 2. OT ratio distribution
    ax2 = axes[1]
    data['ot_ratio'] = data['overtime_amount'] / data['total_amount'].replace(0, 1)

    normal_ratio = data[data['is_anomaly'] == False]['ot_ratio']
    anomaly_ratio = data[data['is_anomaly'] == True]['ot_ratio']

    ax2.hist(normal_ratio, bins=30, alpha=0.7, label='Normal', color='steelblue', density=True)
    ax2.hist(anomaly_ratio, bins=15, alpha=0.7, label='Anomaly', color='crimson', density=True)
    ax2.axvline(x=0.3, color='red', linestyle='--', label='Threshold (30%)')
    ax2.set_xlabel('Overtime Ratio')
    ax2.set_ylabel('Density')
    ax2.set_title('Overtime as % of Total Pay')
    ax2.legend()

    # 3. Department heatmap
    ax3 = axes[2]
    pivot = data.pivot_table(values='is_anomaly', index='department',
                             columns='role', aggfunc='mean', fill_value=0) * 100

    # Only show top roles
    top_roles = data['role'].value_counts().head(6).index
    pivot = pivot[[c for c in pivot.columns if c in top_roles]]

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Anomaly %'})
    ax3.set_title('Anomaly Rate by Dept & Role')
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{save_path}feature_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}feature_analysis.png")


def print_summary_stats(data):
    """Print key statistics"""
    print("\n" + "="*60)
    print("ANOMALY DETECTION SUMMARY")
    print("="*60)

    total = len(data)
    anomalies = data['is_anomaly'].sum()

    print(f"\nDataset: {total:,} transactions")
    print(f"Anomalies: {anomalies} ({anomalies/total*100:.1f}%)")

    print("\nAnomaly Types:")
    for atype, count in data[data['is_anomaly']==True]['anomaly_type'].value_counts().items():
        print(f"  - {atype}: {count}")

    print("\nTop 5 Suspicious Transactions:")
    print("-"*60)

    # Show some actual anomalies
    suspicious = data[data['is_anomaly'] == True].nlargest(5, 'total_amount')
    for _, row in suspicious.iterrows():
        print(f"  {row['transaction_id']} | {row['department']:12} | "
              f"${row['total_amount']:,.0f} | OT: {row['overtime_hours']:.0f}h | "
              f"Type: {row['anomaly_type']}")

    print("\n" + "="*60)


def main():
    print("Loading data...")
    data = load_data()

    print_summary_stats(data)

    print("\nGenerating visualizations...")
    os.makedirs('reports', exist_ok=True)

    plot_anomaly_distribution(data)
    plot_time_series(data)
    plot_feature_analysis(data)

    print("\nDone! Check the reports/ folder for saved charts.")


if __name__ == "__main__":
    main()
