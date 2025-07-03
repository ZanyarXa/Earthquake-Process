# src/neighbor_utils.py

import numpy as np
from sklearn.neighbors import BallTree

def compute_neighbors(df, radius_km=50):
    coords_rad = np.deg2rad(df[['Latitude', 'Longitude']].values)

    tree = BallTree(coords_rad)

    earth_radius_km = 6371.0

    radius_rad = radius_km / earth_radius_km

    ind = tree.query_radius(coords_rad, r=radius_rad)

    num_neighbors = np.array([len(i) - 1 for i in ind])

    return num_neighbors
