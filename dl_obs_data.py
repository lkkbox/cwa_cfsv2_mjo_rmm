#!/usr/bin/env python
import os
import tools.nctools as nct
import tools.timetools as tt
import tools.caltools as ct
import numpy as np


OBS_DIR = "data/obs"


def main() -> None:
    download_bom_rmm()
    bom_rmm_txt_to_nc()
    # download_noaa_olr()
    # download_era5_u(850, 2025)
    # download_era5_u(200, 2025)


def download_noaa_olr() -> None:
    """
    download https://psl.noaa.gov/thredds/fileServer/Datasets/cpc_blended_olr-2.5deg/olr.cbo-2.5deg.day.mean.nc
    """

    url = "https://psl.noaa.gov/thredds/fileServer/Datasets/cpc_blended_olr-2.5deg/olr.cbo-2.5deg.day.mean.nc"
    os.system(f"wget --continue -P {OBS_DIR} '{url}'")


def download_era5_u(level: int, year: int) -> None:
    import cdsapi

    target = f"{OBS_DIR}/era5_u{level}_{year}.nc"

    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["u_component_of_wind"],
        "year": [f"{year}"],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"],
        "pressure_level": [f"{level}"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(target)


def download_bom_rmm() -> None:
    url = "https://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt"
    agent = '-U "Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0"'
    os.system(f"wget {agent} --continue -P {OBS_DIR} '{url}'")


def bom_rmm_txt_to_nc() -> None:
    txtName = "./data/obs/rmm.74toRealtime.txt"
    ncName = "./data/obs/rmm.74toRealtime.nc"

    times: list[float] = []
    rmm1s: list[float] = []
    rmm2s: list[float] = []
    with open(txtName, "rt") as f:
        f.readline()  # skip the headers (2 lines)
        f.readline()
        for i in range(999999999):
            line = f.readline()
            if not line:
                break

            text = line.split()
            # year, month, day, RMM1, RMM2, phase, amplitude.  Missing Value= 1.E36 or 999
            year = int(text[0])
            month = int(text[1])
            day = int(text[2])
            rmm1 = float(text[3])
            rmm2 = float(text[4])

            time = tt.ymd2float(year, month, day)
            times.append(time)
            rmm1s.append(rmm1)
            rmm2s.append(rmm2)

    nct.save(
        ncName,
        {
            "rmm1": rmm1s,
            "time": times,
        },
        overwrite=True,
    )
    nct.save(
        ncName,
        {
            "rmm2": rmm2s,
            "time": times,
        },
        overwrite=True,
    )

    print(f"saved: {ncName}")


if __name__ == "__main__":
    main()
