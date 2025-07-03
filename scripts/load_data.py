# load_data.py

import pandas as pd

def load_earthquake_data():
    return pd.read_csv("dataset/earthquakes.csv")

def load_faults_data():
    return "dataset/gem_active_faults_harmonized.geojson"  # Path only

def load_moho_data():
    return pd.read_csv("dataset/depthtomoho.xyz", delim_whitespace=True, header=None,
                       names=["Longitude", "Latitude", "MohoDepth"])

def load_crust_data():
    return pd.read_csv("dataset/crsthk.xyz", delim_whitespace=True, header=None,
                       names=["Longitude", "Latitude", "CrustThickness"])

def load_engineered_dataset():
    return pd.read_csv("dataset/engineered_dataset.csv")
