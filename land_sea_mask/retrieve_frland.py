import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset(
    "./ceffsurf000_000",
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {"typeOfLevel": "surface", "stepType": "instant"}
    },
)

ds=ds.drop_vars([
    "step",
    "valid_time",
    "surface",
    "latitude",
    "longitude",
    "time",
])

plt.figure()
ds.lsm.plot(add_colorbar=False)
plt.title("Land-sea mask")
plt.savefig("land_sea_mask.png")
plt.close()

ds.lsm.to_zarr("land_sea_mask.zarr", mode="w")
