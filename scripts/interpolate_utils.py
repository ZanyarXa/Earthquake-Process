# src/interpolate_utils.py

import numpy as np
from scipy.interpolate import griddata

def interpolate_feature(source_df, source_col, target_df, lon_col="Longitude", lat_col="Latitude"):
    points = source_df[[lon_col, lat_col]].values
    values = source_df[source_col].values

    interpolated = griddata(points, values, (target_df['Longitude'], target_df['Latitude']), method='linear')

    interpolated_nearest = griddata(points, values, (target_df['Longitude'], target_df['Latitude']), method='nearest')
    interpolated_final = np.where(np.isnan(interpolated), interpolated_nearest, interpolated)

    return interpolated_final
