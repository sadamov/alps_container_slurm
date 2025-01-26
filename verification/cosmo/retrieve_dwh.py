import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from pyproj import Transformer

pyproj.datadir.get_data_dir()  # This will initialize proj properly
# Get conda environment path
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    proj_lib = str(Path(conda_prefix) / "share" / "proj")
    os.environ["PROJ_LIB"] = proj_lib
    pyproj.datadir.set_data_dir(proj_lib)
    print(f"PROJ_LIB set to {proj_lib}")


def parse_station_info(lines):
    stations = {}
    pattern = (
        r"\s*(\d+)\s+"  # Station ID with flexible leading space
        r"([\w\s/-]+?)\s+"  # Name (non-greedy match)
        r"(\d+)\s+m\s+a\.s\.l\."  # Height
        r"\s+(\d+)/\s*(\d+)"  # Coordinates
    )

    for line in lines:
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            station_id, name, height, x, y = match.groups()
            stations[int(station_id)] = {
                "name": name.strip(),
                "height": int(height),
                "x": int(x),
                "y": int(y),
            }
        else:
            print(f"Failed to parse line: '{line}'")

    return stations


def parse_parameter_info(lines):
    params = {}
    pattern = r"(\d+)\s+(.*?)\s+\[(.*?)\]"

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            param_id, description, unit = match.groups()
            params[int(param_id)] = {
                "description": description.strip(),
                "unit": unit,
            }
    return params


def swiss_to_wgs84(x, y):
    """Convert Swiss LV03 coordinates to WGS84 lat/lon."""
    transformer = Transformer.from_crs("EPSG:21781", "EPSG:4326")
    lat, lon = transformer.transform(x, y)
    return lat, lon


def get_cf_names():
    """Map the numeric codes to CF convention names"""
    return {
        "1739": "air_temperature",  # [°C]
        "1741": "relative_humidity",  # [%]
        "1742": "air_pressure",  # QFE [hPa]
        "282": "wind_direction",  # [°]
        "283": "wind_speed",  # [m/s]
        "267": "precipitation",  # [mm]
    }


def create_xarray_dataset(data_df, stations):
    # Get actual station IDs from the data
    data_station_ids = sorted(data_df["STA"].unique())

    # Filter stations dictionary
    stations = {
        sid: stations[sid] for sid in data_station_ids if sid in stations
    }

    # Create coordinates and convert from Swiss to WGS84
    station_ids = list(stations.keys())
    swiss_x = np.array([stations[sid]["x"] for sid in station_ids])
    swiss_y = np.array([stations[sid]["y"] for sid in station_ids])

    # Convert coordinates
    lats, lons = swiss_to_wgs84(swiss_x, swiss_y)

    # Convert time columns to datetime
    data_df["time"] = pd.to_datetime({
        "year": data_df["JAHR"],
        "month": data_df["MO"],
        "day": data_df["TG"],
        "hour": data_df["HH"],
        "minute": data_df["MM"],
    })

    # Get CF convention names
    cf_names = get_cf_names()

    # Create DataArrays for each parameter
    data_vars = {}
    for code, cf_name in cf_names.items():
        if code in data_df.columns:
            filtered_df = data_df[data_df["STA"].isin(station_ids)]
            values = filtered_df.pivot(
                index="time", columns="STA", values=code
            ).values

            data_vars[cf_name] = xr.DataArray(
                data=values,
                dims=["time", "station"],
                coords={
                    "time": filtered_df["time"].unique(),
                    "station": station_ids,
                    "latitude": ("station", lats),
                    "longitude": ("station", lons),
                },
                attrs={"standard_name": cf_name, "original_code": code},
            )

    # Add diagnostic information
    print(f"Total stations in data: {len(data_station_ids)}")
    print(f"Stations with metadata: {len(station_ids)}")
    print(f"Missing stations: {set(data_station_ids) - set(station_ids)}")

    return xr.Dataset(data_vars)


def process_data_to_zarr(filename, output_zarr):
    # Change encoding to latin1 or iso-8859-1
    with open(filename, "r", encoding="latin1") as f:
        lines = f.readlines()

    # Rest of the code remains the same
    station_lines = [
        l
        for l in lines
        if re.match(r"\s+\d+\s+[\w\s/-]+?\s+\d+\s+m\s+a\.s\.l\.", l)
    ]
    param_lines = [l for l in lines if re.match(r"\s+\d+\s+.*?\[.*?\]", l)]

    stations = parse_station_info(station_lines)
    parameters = parse_parameter_info(param_lines)

    data_start = (
        next(i for i, l in enumerate(lines) if "STA JAHR MO TG HH MM" in l) + 1
    )

    # Updated read_csv with sep instead of delim_whitespace
    data = pd.read_csv(
        filename,
        skiprows=data_start,
        sep=r"\s+",  # This replaces delim_whitespace=True
        encoding="latin1",
        names=[
            "STA",
            "JAHR",
            "MO",
            "TG",
            "HH",
            "MM",
            "1739",
            "1741",
            "1742",
            "282",
            "283",
            "267",
        ],
    )

    ds = create_xarray_dataset(data, stations)
    ds.attrs["description"] = "Weather station data"
    ds.attrs["parameters"] = parameters
    ds.to_zarr(output_zarr, mode="w")


process_data_to_zarr("complete.dat", "observations.zarr")
