import requests
import pyproj


def transform_coord(proj1, proj2, lon, lat):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)
    # Convert coordinates
    return pyproj.transform(proj1, proj2, lon, lat)

def get_granule_urls(params):
    base_url = 'https://nsidc.org/apps/itslive-search/velocities/urls'
    resp = requests.get(base_url, params=params, verify=False)
    return resp.json()
