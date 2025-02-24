#!/usr/bin/env python3
"""
MOQUAS (Model Quality Score for Switzerland) Implementation
Simplified for ML-Model Hyperparameter Tuning
Based on MeteoSwiss documentation v1.0.1

Author: [Simon Adamov]
Email: [simon.adamov@meteoswiss.ch]
Date: [2025-02-08]
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

# ===============================
# CONSTANTS AND CONFIGURATIONS
# ===============================

PATH_GROUND_TRUTH = "/capstor/store/cscs/swissai/a01/sadamov/cosmo_sample.zarr"
PATH_MODEL = "/capstor/store/cscs/swissai/a01/sadamov/cosmo_sample.zarr"


@dataclass
class MOQUASConfig:
    """Configuration parameters for MOQUAS calculation"""

    # Weights for different components (must sum to 100)
    cloud_weight: float = 30.0
    precip_weight: float = 30.0
    temp_weight: float = 30.0
    wind_weight: float = 10.0

    # Thresholds for categorical scores
    cloud_threshold: float = 2.5  # octas
    precip_threshold: float = 0.1  # mm/12h

    # Temperature tolerance/utility thresholds
    temp_tolerance: float = 0.1  # °C
    temp_utility: float = 6.0  # °C

    # Wind speed tolerance/utility thresholds (for normalized MAE)
    wind_tolerance: float = 0.1  # fraction
    wind_utility: float = 2.0  # factor


class MOQUASCalculator:
    """Main class for calculating MOQUAS scores"""

    def __init__(self, config: MOQUASConfig = None):
        self.config = config or MOQUASConfig()

    def compute_ets(
        self, forecast: xr.DataArray, obs: xr.DataArray, threshold: float
    ) -> float:
        """
        Compute Equitable Threat Score (ETS) and rescale it.

        ETSrescaled = ETS/2 + 0.5

        This rescales the score for performance "only as good as sample climatology"
        from 0 in ETS to 0.5 in ETSrescaled, while keeping maximum at 1.0
        """
        # Convert to binary using threshold
        f_binary = (forecast > threshold).astype(int)
        o_binary = (obs > threshold).astype(int)

        # Ensure arrays are computed before operations
        try:
            # Compute contingency table
            hits = float(((f_binary == 1) & (o_binary == 1)).sum().compute())
            false_alarms = float(
                ((f_binary == 1) & (o_binary == 0)).sum().compute()
            )
            misses = float(((f_binary == 0) & (o_binary == 1)).sum().compute())
            correct_zeros = float(
                ((f_binary == 0) & (o_binary == 0)).sum().compute()
            )

            total = hits + false_alarms + misses + correct_zeros

            # Calculate hits due to chance (hr)
            hits_random = ((hits + false_alarms) * (hits + misses)) / total

            # Calculate original ETS score
            denominator = hits - hits_random + false_alarms + misses
            if denominator == 0:
                ets = 0.0 if total > 0 else 1.0
            else:
                ets = (hits - hits_random) / denominator

            # Apply MeteoSwiss rescaling: ETSrescaled = ETS/2 + 0.5
            ets_rescaled = ets / 2.0 + 0.5

            return max(0.0, min(1.0, ets_rescaled))
        except Exception as e:
            print(f"Error computing ETS: {e}")
            return 0.0

    def compute_mae_skill(
        self,
        forecast: xr.DataArray,
        obs: xr.DataArray,
        tolerance: float,
        utility: float,
    ) -> float:
        """
        Compute skill score from MAE using COMFORT-style scaling
        """
        try:
            mae = float(abs(forecast - obs).mean().compute())
            if mae <= tolerance:
                return 1.0
            elif mae >= utility:
                return 0.0
            else:
                return 1.0 - (mae - tolerance) / (utility - tolerance)
        except Exception as e:
            print(f"Error computing MAE skill: {e}")
            return 0.0

    def compute_moquas(
        self, data: Dict[str, xr.DataArray]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute MOQUAS score directly from a dictionary of DataArrays
        """
        # Calculate cloud cover ETS
        cloud_ets = self.compute_ets(
            data["cloud_forecast"],
            data["cloud_obs"],
            self.config.cloud_threshold,
        )

        # Calculate precipitation ETS
        precip_ets = self.compute_ets(
            data["precip_forecast"],
            data["precip_obs"],
            self.config.precip_threshold,
        )

        # Calculate temperature MAE skill
        temp_skill = self.compute_mae_skill(
            data["temp_forecast"],
            data["temp_obs"],
            self.config.temp_tolerance,
            self.config.temp_utility,
        )

        # Calculate normalized wind MAE skill
        try:
            wind_mae = float(
                abs(data["wind_forecast"] - data["wind_obs"]).mean().compute()
            )
            mean_wind = float(data["wind_obs"].mean().compute())
            if mean_wind > 0:
                norm_wind_mae = wind_mae / mean_wind
            else:
                norm_wind_mae = wind_mae

            wind_skill = self.compute_mae_skill(
                data["wind_forecast"],
                data["wind_obs"],
                self.config.wind_tolerance,
                self.config.wind_utility,
            )
        except Exception as e:
            print(f"Error computing wind skill: {e}")
            wind_skill = 0.0

        # Combine partial scores into MOQUAS
        partial_scores = {
            "cloud": cloud_ets,
            "precip": precip_ets,
            "temp": temp_skill,
            "wind": wind_skill,
        }

        # Calculate final MOQUAS score (weights as per documentation)
        moquas = (
            30.0 * cloud_ets  # 30% weight for cloud cover
            + 30.0 * precip_ets  # 30% weight for precipitation
            + 30.0 * temp_skill  # 30% weight for temperature
            + 10.0 * wind_skill  # 10% weight for wind speed
        )

        return moquas, partial_scores

    def prepare_cosmo_data(
        self, forecast: xr.Dataset, obs: xr.Dataset
    ) -> Dict[str, xr.DataArray]:
        """
        Prepare COSMO model data for MOQUAS calculation

        Args:
            forecast: Forecast Dataset from COSMO model
            obs: Observation/Ground truth Dataset from COSMO model
        """

        # Calculate wind speed from U and V components
        def compute_wind_speed(ds: xr.Dataset) -> xr.DataArray:
            return np.sqrt(ds.U_10M**2 + ds.V_10M**2)

        # Convert radiation to cloud cover proxy (simplified)
        def radiation_to_cloud(ds: xr.Dataset) -> xr.DataArray:
            # ATHB_S is longwave radiation - higher values indicate more clouds
            # First compute over non-NaN values
            athb = ds.ATHB_S
            # Compute min/max over valid data points
            min_rad = float(athb.min())
            max_rad = float(athb.max())

            if min_rad == max_rad:
                return xr.full_like(
                    athb, 4.0
                )  # Return middle value if no variation

            return 8 * (athb - min_rad) / (max_rad - min_rad)

        return {
            "cloud_forecast": radiation_to_cloud(forecast),
            "cloud_obs": radiation_to_cloud(obs),
            "precip_forecast": forecast.TOT_PREC,
            "precip_obs": obs.TOT_PREC,
            "temp_forecast": forecast.T_2M,
            "temp_obs": obs.T_2M,
            "wind_forecast": compute_wind_speed(forecast),
            "wind_obs": compute_wind_speed(obs),
        }


# Example usage
if __name__ == "__main__":
    # Load COSMO data with proper time selection
    ds_gt = xr.open_zarr(PATH_GROUND_TRUTH)
    ds_ml = xr.open_zarr(PATH_MODEL)

    ds_gt = ds_gt.isel(time=slice(0, -1))
    ds_ml = ds_ml.isel(time=slice(1, None))
    ds_ml["time"] = ds_gt["time"]

    calculator = MOQUASCalculator()
    sample_data = calculator.prepare_cosmo_data(ds_ml, ds_gt)
    moquas_score, component_scores = calculator.compute_moquas(sample_data)

    # Print results
    print("\nMOQUAS Score for COSMO Data:")
    print("-" * 35)
    print(f"Overall MOQUAS: {moquas_score:.3f}")
    print("\nComponent Scores:")
    for component, score in component_scores.items():
        print(f"{component.capitalize():>10}: {score:.3f}")
