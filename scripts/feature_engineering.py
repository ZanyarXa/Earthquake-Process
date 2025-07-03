import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from scripts.load_data import load_earthquake_data, load_faults_data, load_moho_data, load_crust_data
from scripts.distance_utils import compute_distance_km
from scripts.interpolate_utils import interpolate_feature
from scripts.neighbor_utils import compute_neighbors

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

def run_feature_engineering():
    # --- بارگذاری داده‌های اولیه ---
    earthquake_df = load_earthquake_data()
    faults_geojson_path = load_faults_data()

    # تبدیل به GeoDataFrame برای پردازش جغرافیایی
    geometry = [Point(lon, lat) for lon, lat in zip(earthquake_df["Longitude"], earthquake_df["Latitude"])]
    gdf_quakes = gpd.GeoDataFrame(earthquake_df, geometry=geometry, crs="EPSG:4326")

    gdf_faults = gpd.read_file(faults_geojson_path).to_crs("EPSG:4326")
    fault_sindex = gdf_faults.sindex

    # --- یافتن نزدیک‌ترین گسل برای هر زلزله ---
    nearest_fault_ids = []
    nearest_fault_names = []
    nearest_points = []
    distances_km = []
    fault_point_lats = []
    fault_point_lons = []

    search_radii_km = [10, 25, 50, 100, 200, 500]

    print(f"Processing {len(gdf_quakes)} earthquakes...")

    for i, quake_point in enumerate(gdf_quakes.geometry):
        found = False
        for radius_km in search_radii_km:
            radius_deg = radius_km / 111.0
            buffer = quake_point.buffer(radius_deg)
            possible_faults_idx = list(fault_sindex.intersection(buffer.bounds))

            if possible_faults_idx:
                possible_faults = gdf_faults.iloc[possible_faults_idx]
                min_dist_km = float("inf")
                best_idx = None
                best_point = None

                for idx, fault_geom in possible_faults.geometry.items():
                    p_on_fault = fault_geom.interpolate(fault_geom.project(quake_point))
                    dist_km = compute_distance_km(quake_point, p_on_fault)

                    if dist_km < min_dist_km:
                        min_dist_km = dist_km
                        best_idx = idx
                        best_point = p_on_fault

                if best_idx is not None:
                    nearest_fault_ids.append(best_idx)
                    nearest_fault_names.append(gdf_faults.loc[best_idx, "name"])
                    nearest_points.append(best_point)
                    distances_km.append(min_dist_km)
                    fault_point_lats.append(best_point.y)
                    fault_point_lons.append(best_point.x)
                    found = True
                    break

        if not found:
            nearest_fault_ids.append(None)
            nearest_fault_names.append(None)
            nearest_points.append(None)
            distances_km.append(None)
            fault_point_lats.append(None)
            fault_point_lons.append(None)

        if (i + 1) % 100 == 0 or (i + 1) == len(gdf_quakes):
            print(f"Processed {i + 1}/{len(gdf_quakes)} earthquakes")

    gdf_quakes["nearest_fault_id"] = nearest_fault_ids
    gdf_quakes["nearest_fault_name"] = nearest_fault_names
    gdf_quakes["distance_to_fault_km"] = distances_km
    gdf_quakes["fault_point_lat"] = fault_point_lats
    gdf_quakes["fault_point_lon"] = fault_point_lons

    # --- بارگذاری و اضافه کردن داده‌های ژئوفیزیکی (موهو و پوسته) ---
    moho_df = load_moho_data()
    print(f"Loaded Moho data: {moho_df.shape[0]} points")
    crust_df = load_crust_data()
    print(f"Loaded Crustal Thickness data: {crust_df.shape[0]} points")

    gdf_quakes['moho_depth'] = interpolate_feature(moho_df, 'MohoDepth', gdf_quakes)
    print("Moho depth added.")
    gdf_quakes['crustal_thickness'] = interpolate_feature(crust_df, 'CrustThickness', gdf_quakes)
    print("Crustal thickness added.")

    # --- محاسبه تعداد همسایگان در شعاع 50 کیلومتر ---
    gdf_quakes['num_neighbors_50km'] = compute_neighbors(gdf_quakes, radius_km=50)
    print("50km neighbors added.")

    # --- استخراج ویژگی‌های زمانی از ستون 'time' ---
    gdf_quakes = pd.read_csv('dataset/earthquakes.csv')
    gdf_quakes = gdf_quakes.merge(
        gdf_quakes[['Year', 'Month', 'Quarter', 'Day']],
        left_index=True,
        right_index=True
    )

    # --- دسته‌بندی عمق ---
    gdf_quakes['depth_category'] = pd.cut(gdf_quakes['Depth'], bins=[-1, 70, 300, 700], labels=['Shallow', 'Intermediate', 'Deep'])

    # --- ترکیب ستون‌های متنی ---
    gdf_quakes['location_combined'] = gdf_quakes['Location Source'] + '_' + gdf_quakes['Source']

    # --- کدگذاری دسته‌ای با One-hot برای 'type' و 'magnitude_type' ---
    gdf_quakes = pd.get_dummies(gdf_quakes, columns=['Type', 'Magnitude Type'], drop_first=True)


    # --- دسته‌بندی بزرگی زلزله ---
    gdf_quakes['magnitude_category'] = gdf_quakes['Magnitude'].apply(categorize_magnitude)

    # --- مدیریت مقادیر پرت ---
    gdf_quakes = handle_outliers(gdf_quakes, 'Magnitude')
    gdf_quakes = handle_outliers(gdf_quakes, 'Depth')

    # --- تبدیل لگاریتمی ---
    gdf_quakes['log_magnitude'] = np.log1p(gdf_quakes['Magnitude'])
    gdf_quakes['log_depth'] = np.log1p(gdf_quakes['Depth'])

   
    # --- ویژگی‌های تعاملی ---
    gdf_quakes['depth_to_magnitude'] = gdf_quakes['Depth'] / (gdf_quakes['Magnitude'] + 1)
    gdf_quakes['magnitude_depth_interaction'] = gdf_quakes['Magnitude'] * gdf_quakes['Depth']

    # --- ذخیره نتایج ---
    output_csv = "engineered_dataset_combined.csv"
    gdf_quakes.to_csv(output_csv, index=False)
    print(f"Feature engineering completed and saved to {output_csv}")

if __name__ == "__main__":
    run_feature_engineering()

