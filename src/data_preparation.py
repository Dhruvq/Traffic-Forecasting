import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io

def main():
    print("Creating directories...")
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../images', exist_ok=True)

    print("Fetching Metro Interstate Traffic Volume dataset ZIP...")
    try:
        url = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
        response = requests.get(url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open('Metro_Interstate_Traffic_Volume.csv.gz') as f:
                df = pd.read_csv(f, compression='gzip')
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return

    print("Saving raw data to CSV...")
    df.to_csv('../data/traffic_data.csv', index=False)

    # Preprocessing
    df['holiday'] = df['holiday'].fillna('None')
    print("Missing values after fill:\n", df.isnull().sum())

    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)

    df = df.drop(columns=['weather_description'])

    df['hour']        = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek   # 0=Mon, 6=Sun
    df['month']       = df['date_time'].dt.month        # 1–12

    # Cyclical encoding for temporal features.
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    print(f"Total samples: {len(df)}")

    # 80 / 10 / 10 time-ordered split
    N = len(df)
    train_end = int(N * 0.80)
    val_end   = int(N * 0.90)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df  = df.iloc[val_end:].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    df.to_csv('../data/preprocessed_traffic_data.csv', index=False)
    train_df.to_csv('../data/train.csv', index=False)
    val_df.to_csv('../data/val.csv', index=False)
    test_df.to_csv('../data/test.csv', index=False)

    split_indices = {"train_end": train_end, "val_end": val_end, "total": N}
    with open('../data/split_indices.json', 'w') as f:
        json.dump(split_indices, f, indent=2)

    print("Generating EDA plots...")
    sns.set_theme(style="whitegrid")

    # Plot 1: Traffic volume distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['traffic_volume'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Hourly Traffic Volume', fontsize=14)
    plt.xlabel('Traffic Volume (vehicles/hour)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('../images/traffic_volume_dist.png', dpi=150)
    plt.close()

    # Plot 2: Average traffic by hour of day
    avg_by_hour = df.groupby('hour')['traffic_volume'].mean()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=avg_by_hour.index, y=avg_by_hour.values,
                 marker='o', linewidth=2, color='coral')
    plt.title('Average Traffic Volume by Hour of Day', fontsize=14)
    plt.xlabel('Hour of Day (0–23)')
    plt.ylabel('Average Traffic Volume (vehicles/hour)')
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig('../images/avg_hourly_traffic.png', dpi=150)
    plt.close()

    # Plot 3: Average traffic by day of week
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    avg_by_dow = df.groupby('day_of_week')['traffic_volume'].mean()
    plt.figure(figsize=(9, 5))
    plt.bar(day_labels, avg_by_dow.values,
            color=['steelblue']*5 + ['tomato']*2)
    plt.title('Average Traffic Volume by Day of Week', fontsize=14)
    plt.xlabel('Day of Week')
    plt.ylabel('Average Traffic Volume (vehicles/hour)')
    plt.tight_layout()
    plt.savefig('../images/avg_by_dow.png', dpi=150)
    plt.close()

    # Plot 4: Correlation heatmap (numeric features + target)
    numeric_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all',
                    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                    'month_sin', 'month_cos', 'traffic_volume']
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5)
    plt.title('Pearson Correlation — Numeric Features vs Target', fontsize=13)
    plt.tight_layout()
    plt.savefig('../images/correlation_heatmap.png', dpi=150)
    plt.close()

    # Plot 5: Monthly traffic boxplot (seasonal spread)
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    plt.figure(figsize=(12, 6))
    df.boxplot(column='traffic_volume', by='month',
               notch=False, patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='navy', linewidth=2),
               flierprops=dict(markersize=2, alpha=0.3))
    plt.xticks(range(1, 13), month_labels)
    plt.title('Traffic Volume Distribution by Month', fontsize=14)
    plt.suptitle('')
    plt.xlabel('Month')
    plt.ylabel('Traffic Volume (vehicles/hour)')
    plt.tight_layout()
    plt.savefig('../images/monthly_traffic_boxplot.png', dpi=150)
    plt.close()

    # Plot 6: Cyclical encoding demo
    hours = np.arange(24)
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].bar(hours, hours, color='steelblue', alpha=0.8)
    axes[0].set_title('Raw Integer Encoding: Hour 23 and Hour 0 are 23 units apart', fontsize=11)
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Raw Value')
    axes[0].set_xticks(hours)

    axes[1].plot(hours, hour_sin, marker='o', label='hour_sin', color='coral')
    axes[1].plot(hours, hour_cos, marker='s', label='hour_cos', color='teal')
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=23, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('Cyclical Encoding: Hour 23 and Hour 0 are close on the unit circle', fontsize=11)
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Encoded Value')
    axes[1].set_xticks(hours)
    axes[1].legend()

    plt.suptitle('Why Cyclical Encoding?', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/cyclical_encoding_demo.png', dpi=150)
    plt.close()

    print("Data preparation complete.")
    print(f"  Plots saved to ../images/ ({6} plots)")
    print(f"  Data saved to ../data/ (train/val/test CSVs + split_indices.json)")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
