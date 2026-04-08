import os
import pandas as pd
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

    print("Saving to CSV...")
    df.to_csv('../data/traffic_data.csv', index=False)
    
    print("Data exploration and preprocessing...")
    # Fill missing holiday values with 'None'
    df['holiday'] = df['holiday'].fillna('None')
    
    print("Missing values in dataset:\n", df.isnull().sum())
    
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Sort by datetime to respect time-series nature before splitting down the line
    df = df.sort_values('date_time').reset_index(drop=True)
    
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    
    print("Saving preprocessed data to CSV...")
    # Drop columns that are text-heavy (e.g., 'weather_description' is too detailed compared to 'weather_main')
    df_preprocessed = df.drop(columns=['weather_description'])
    df_preprocessed.to_csv('../data/preprocessed_traffic_data.csv', index=False)
    
    print("Generating EDA plots...")
    # Setup aesthetic
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Traffic volume distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['traffic_volume'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Hourly Traffic Volume')
    plt.xlabel('Traffic Volume (vehicles)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('../images/traffic_volume_dist.png')
    plt.close()
    
    # Plot 2: Average Traffic volume by hour
    avg_by_hour = df.groupby('hour')['traffic_volume'].mean()
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=avg_by_hour.index, y=avg_by_hour.values, marker='o', linewidth=2, color='coral')
    plt.title('Average Traffic Volume by Hour of Day')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Average Traffic Volume')
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig('../images/avg_hourly_traffic.png')
    plt.close()

    print("Data preparation complete.")

if __name__ == '__main__':
    # Adjust working directory to the script's location so relative paths work nicely
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
