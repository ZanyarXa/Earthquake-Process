# preprocess.py

def preprocess_for_modeling(df):
    df_model = df[["distance_to_fault_km", "Depth", "moho_depth", "crustal_thickness", "num_neighbors_50km", "Magnitude"]].dropna()
    return df_model
