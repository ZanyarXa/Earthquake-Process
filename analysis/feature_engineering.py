import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def categorize_magnitude(mag):
    if mag < 5:
        return 'Low'
    elif mag < 6:
        return 'Moderate'
    elif mag < 7:
        return 'Strong'
    else:
        return 'Severe'

def handle_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

def feature_engineering():
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR / '..' / 'data' / 'earthquakes_cleaned.csv'
    df = pd.read_csv(csv_path)

    df['year'] = pd.to_datetime(df['time']).dt.year
    df['month'] = pd.to_datetime(df['time']).dt.month
    df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
    df['quarter'] = pd.to_datetime(df['time']).dt.quarter

    df['depth_category'] = pd.cut(df['depth'], bins=[-1, 70, 300, 700], labels=['Shallow', 'Intermediate', 'Deep'])

    df['location_combined'] = df['location_source'] + '_' + df['source']

    df = pd.get_dummies(df, columns=['type', 'magnitude_type'], drop_first=True)

    df['yearly_avg_magnitude'] = df.groupby('year')['magnitude'].transform('mean')

    df['magnitude_category'] = df['magnitude'].apply(categorize_magnitude)

    df = handle_outliers(df, 'magnitude')
    df = handle_outliers(df, 'depth')

    df['log_magnitude'] = np.log1p(df['magnitude'])
    df['log_depth'] = np.log1p(df['depth'])

    numeric_cols = ['magnitude', 'depth', 'yearly_avg_magnitude']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df['depth_to_magnitude'] = df['depth'] / (df['magnitude'] + 1)
    df['magnitude_depth_interaction'] = df['magnitude'] * df['depth']

    df['depth_category'] = pd.cut(df['depth'], bins=[-1, 70, 300, 700], labels=['Shallow', 'Intermediate', 'Deep'])

    output_path_csv = BASE_DIR / '..' / 'data' / 'engineered_features_cleaned.csv'
    df.to_csv(output_path_csv, index=False)
    output_path_pickle = BASE_DIR / '..' / 'data' / 'engineered_features_cleaned.pkl'
    df.to_pickle(output_path_pickle)

    print("Feature engineering and advanced preprocessing completed.")

if __name__ == "__main__":
    feature_engineering()

