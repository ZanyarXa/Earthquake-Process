import pandas as pd
from pathlib import Path

def preprocess_data():
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR / '..' / 'data' / 'earthquakes.csv'
    df = pd.read_csv(csv_path)


    df = df[[
        'ID', 'Time', 'Latitude', 'Longitude', 'Depth',
        'Magnitude', 'Magnitude Type', 'Type',
        'Location Source', 'Source', 'Magnitude Source', 'Status'
    ]]

    df.columns = [
        'id', 'time', 'latitude', 'longitude', 'depth',
        'magnitude', 'magnitude_type', 'type',
        'location_source', 'source', 'magnitude_source', 'status'
    ]

    df = df.drop_duplicates(subset=['id'])
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'magnitude'])

    df['magnitude_type'] = df['magnitude_type'].fillna('Unknown')
    df['type'] = df['type'].fillna('Unknown')
    df['location_source'] = df['location_source'].fillna('Unknown')
    df['source'] = df['source'].fillna('Unknown')
    df['magnitude_source'] = df['magnitude_source'].fillna('Unknown')
    df['status'] = df['status'].fillna('Unknown')

    df = df[df['depth'] >= 0]
    df = df[(df['magnitude'] >= 0) & (df['magnitude'] <= 10)]

    output_path = BASE_DIR / '..' / 'data' / 'earthquakes_cleaned.csv'
    df.to_csv(output_path, index=False)


    return df

