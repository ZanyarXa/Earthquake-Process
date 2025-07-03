# src/distance_utils.py

from pyproj import Geod

geod = Geod(ellps="WGS84")

def compute_distance_km(p1, p2):
    _, _, dist = geod.inv(p1.x, p1.y, p2.x, p2.y)
    return dist / 1000.0
