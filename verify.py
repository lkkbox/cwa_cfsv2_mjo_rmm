#!/usr/bin/env python
from typing import Literal
from tools import timetools as tt
from tools import nctools as nct
from tools import caltools as ct
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")
FIG_DIR = "./figs/verify"


NUM_LEADS = 40
ENS_NAME = ["m000", "m001", "m002", "mavg"]
COLORS = [
    (0.5, 0, 0),
    (1, 0, 0),
    (1, 0.5, 0.5),
    (0, 0.8, 0.5),
]

LON = np.r_[0:360:2.5]


clevs_total = {
    "u200": ct.mirror([0, 2, 4, 8, 16]),
    "u850": ct.mirror([0, 1, 2, 4, 8]),
    "olr": [170, 190, 210, 230, 250, 270, 290, 310],
}

clevs_anom = {
    "u200": ct.mirror([0, 2, 4, 8, 16]),
    "u850": ct.mirror([0, 1, 2, 4, 8]),
    "olr": ct.mirror([0, 15, 30, 45, 60, 75]),
}


def main() -> None:
    initDate = tt.ymd2float(2026, 1, 14)
    ensName = "m000"
    modTime, modRmm1, modRmm2, modAmp, modPhase = read_computed_rmms_analysis(
        initDate, ensName
    )

    bomTime, bomRmm1, bomRmm2 = read_bom_rmm(modTime[0], modTime[-1])

    modOverlap = np.isin(modTime, bomTime)
    bomOverlap = np.isin(bomTime, modTime)

    modTime = modTime[modOverlap]
    modRmm1 = modRmm1[modOverlap]
    modRmm2 = modRmm2[modOverlap]
    modAmp = modAmp[modOverlap]
    modPhase = modPhase[modOverlap]

    bomTime = bomTime[bomOverlap]
    bomRmm1 = bomRmm1[bomOverlap]
    bomRmm2 = bomRmm2[bomOverlap]

    fig = plt.figure()
    for iax in range(2):
        if iax == 0:
            x1, y1 = bomTime, bomRmm1
            x2, y2 = modTime, modRmm1
            dataName = "rmm1"

        elif iax == 1:
            x1, y1 = bomTime, bomRmm2
            x2, y2 = modTime, modRmm2
            dataName = "rmm2"

        ax = fig.add_subplot(2, 1, iax + 1)

        ax.plot(x1, y1, label="BOM")
        ax.plot(x2, y2, label="Model Analysis")
        ax.legend()

        timeTicks = [
            t for t in x1 if tt.day(float(t)) in [1, 6, 11, 16, 21, 26]
        ]
        timeTickLabels = [tt.float2format(t, "%m%d") for t in timeTicks]
        ax.set_xticks(timeTicks)
        ax.set_xticklabels(timeTickLabels)

    figName = f"{FIG_DIR}/rmm_time_series.png"
    fig.savefig(figName)
    print(f"saved: {figName}")


def read_bom_rmm(
    dateStart: float, dateEnd: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """return time, rmm1, rmm2"""
    path = "./data/obs/rmm.74toRealtime.nc"
    rmm1, (time,) = nct.ncreadByDimRange(path, "rmm1", [[dateStart, dateEnd]])
    rmm2, (time,) = nct.ncreadByDimRange(path, "rmm2", [[dateStart, dateEnd]])
    return time, rmm1, rmm2


def read_computed_rmms_analysis(
    initDate: float, ensName: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    modTime, modRmm1, modRmm2, modAmp, modPhase = (
        read_computed_rmms_analysis_forecast(initDate, ensName)
    )

    isAnalysis = modTime <= initDate
    modTime = modTime[isAnalysis]
    modRmm1 = modRmm1[isAnalysis]
    modRmm2 = modRmm2[isAnalysis]
    modAmp = modAmp[isAnalysis]
    modPhase = modPhase[isAnalysis]

    return modTime, modRmm1, modRmm2, modAmp, modPhase


def read_computed_rmms_analysis_forecast(
    initDate: float, ensName: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """return times, rmm1s, rmm2s, amps, phases"""
    path = tt.float2format(
        initDate, f"./data/output/%Y/MJORMM_R60_%Y%m%d_{ensName}.txt"
    )

    times: list[float] = []
    rmm1s: list[float] = []
    rmm2s: list[float] = []
    amps: list[float] = []
    phases: int[float] = []

    with open(path, "rt") as f:
        f.readline()  # skip the header
        # YYYY mm dd   RMM1    RMM2      AMP Phase
        for i in range(999999):
            line = f.readline()
            if not line:
                break

            text = line.split()

            year = int(text[0])
            month = int(text[1])
            day = int(text[2])

            rmm1 = float(text[3])
            rmm2 = float(text[4])

            amp = float(text[5])
            phase = int(text[6])

            time = tt.ymd2float(year, month, day)

            times.append(time)
            rmm1s.append(rmm1)
            rmm2s.append(rmm2)
            amps.append(amp)
            phases.append(phase)

    return (
        np.array(times),
        np.array(rmm1s),
        np.array(rmm2s),
        np.array(amps),
        np.array(phases),
    )


def run_draw_compare():
    # setup
    validDateStart = tt.ymd2float(2025, 10, 1)
    validDateEnd = tt.ymd2float(2025, 12, 30)
    lead = 1

    for varName in ["olr", "u200", "u850"]:
        run(varName, validDateStart, validDateEnd, lead)


def run(
    varName: str,
    validDateStart: float,
    validDateEnd: float,
    lead: int,
) -> None:
    # create the valid dates array
    validDates = np.r_[validDateStart : validDateEnd + 1]

    # compute the init dates
    initDateStart = validDateStart - lead
    initDateEnd = validDateEnd - lead
    initDates = validDates - lead

    # read model data
    mod_clim = read_mod_clim(varName, lead, initDates)
    mod_total = read_mod_total(varName, lead, initDates)
    mod_anom = mod_total - mod_clim

    # read obs data
    obs_total = read_obs_total(varName, validDateStart, validDateEnd)
    obs_clim = read_obs_clim(varName)
    iDayOfClim = [int(tt.dayOfClim(d) - 1) for d in validDates]
    obs_anom = obs_total - obs_clim[iDayOfClim, :]

    # draw
    draw_compare(
        f"{varName}_total.png",
        mod_total,
        obs_total,
        validDates,
        "total",
        varName,
        lead,
    )
    draw_compare(
        f"{varName}_anom.png",
        mod_anom,
        obs_anom,
        validDates,
        "anom",
        varName,
        lead,
    )
    # draw_compare(
    #     f"{varName}_clim.png",
    #     mod_clim,
    #     obs_clim,
    #     validDates,
    #     "clim",
    #     varName,
    #     lead,
    # )


def draw_compare(
    figName: str,
    mod: np.ndarray,
    obs: np.ndarray,
    time: np.ndarray,
    statics: Literal["total", "anom", "clim"],
    varName: str,
    lead: int,
) -> None:
    if statics == ["total", "clim"]:
        clevs_main = clevs_total[varName]
    else:
        clevs_main = clevs_anom[varName]
    clevs_diff = [c / 2 for c in clevs_anom[varName]]

    fig = plt.figure()
    ncols = 3
    for icol in range(ncols):
        if icol == 0:
            z = obs
            dataName = "obs"
            clevs = clevs_main

        elif icol == 1:
            z = mod
            dataName = f"mod ({lead=})"
            clevs = clevs_main

        elif icol == 2:
            z = mod - obs
            dataName = "mod - obs"
            clevs = clevs_diff

        iax = icol
        ax = fig.add_subplot(1, ncols, iax + 1)

        clevels = clevs_anom

        cax = ax.contourf(
            LON, time, z, levels=clevs, cmap="coolwarm", extend="both"
        )
        fig.colorbar(cax, orientation="horizontal")
        # ax.clabel(cax)

        # features
        ax.set_title(dataName)

        timeTicks = [t for t in time if tt.day(t) in [1, 5, 10, 15, 20, 25]]
        timeTickLabels = [tt.float2format(t, "%Y%m%d") for t in timeTicks]
        ax.set_yticks(timeTicks)
        if iax == 0:
            ax.set_yticklabels(timeTickLabels)
        else:
            ax.set_yticklabels([])

    fig.suptitle(f"{varName} {statics}")
    fig.savefig(f"{FIG_DIR}/{figName}")


def read_mod_total(
    varName: str,
    lead: int,
    initDates: list[float],
) -> np.ndarray:
    lonBnds = [None, None]
    latBnds = [-15, 15]
    leadBnds = [lead, lead]
    dimBnds = [leadBnds, latBnds, lonBnds]

    paths = [
        tt.float2format(
            date,
            f"data/daymean/%Y/%y%m%d_{varName}.nc",
        )
        for date in initDates
    ]

    datas = []
    for path in paths:
        data, dims = nct.ncreadByDimRange(
            path, varName, dimBnds, decodeTime=False
        )

        datas.append(data)

    datas = mermean_zonint(datas, dims[-2], dims[-1])
    datas = np.squeeze(datas)
    return datas


def read_mod_clim(
    varName: str,
    lead: int,
    initDates: list[float],
) -> np.ndarray:
    lonBnds = [None, None]
    latBnds = [-15, 15]
    leadBnds = [lead, lead]
    dimBnds = [leadBnds, latBnds, lonBnds]

    paths = [
        tt.float2format(
            date,
            f"data/clim_mod/{varName}/global_daily_2p5_{varName}_%m%d_1991_2020_3harm.nc",
        )
        for date in initDates
    ]

    datas = []
    for path in paths:
        data, dims = nct.ncreadByDimRange(
            path, varName, dimBnds, decodeTime=False
        )

        datas.append(data)

    datas = mermean_zonint(datas, dims[-2], dims[-1])
    datas = np.squeeze(datas)
    return datas


def read_obs_clim(
    varName: str,
) -> np.ndarray:
    lonBnds = [0, 360]
    latBnds = [-15, 15]
    timeBnds = [None, None]
    dimBnds = [timeBnds, latBnds, lonBnds]

    path = {
        "u200": "data/clim_obs/obs_u200_clim_2p5.nc",
        "u850": "data/clim_obs/obs_u850_clim_2p5.nc",
        "olr": "data/clim_obs/obs_olr_clim_2p5.nc",
    }[varName]

    ncVarName = {
        "u200": "u200",
        "u850": "u850",
        "olr": "olr",
    }[varName]

    print(f"reading {path}")
    data, dims = nct.ncreadByDimRange(path, ncVarName, dimBnds)
    data = np.squeeze(data)

    data = mermean_zonint(data, dims[-2], dims[-1])
    print(f"clim {varName} shape = {data.shape}")
    return data


def read_obs_total(
    varName: str,
    dateStart: float,
    dateEnd: float,
) -> np.ndarray:
    lonBnds = [0, 360]
    latBnds = [-15, 15]
    levBnds = [None, None]
    timeBnds = [dateStart, dateEnd]

    path = {
        "u200": "data/obs/era5_u200_2025.nc",
        "u850": "data/obs/era5_u850_2025.nc",
        "olr": "data/obs/olr.cbo-2.5deg.day.mean.nc",
    }[varName]

    ncVarName = {
        "u200": "u",
        "u850": "u",
        "olr": "olr",
    }[varName]

    dimBnds = {
        "u200": [timeBnds, levBnds, latBnds, lonBnds],
        "u850": [timeBnds, levBnds, latBnds, lonBnds],
        "olr": [timeBnds, latBnds, lonBnds],
    }[varName]

    print(f"reading {path}")
    data, dims = nct.ncreadByDimRange(path, ncVarName, dimBnds)
    data = np.squeeze(data)

    data = mermean_zonint(data, dims[-2], dims[-1])
    return data


def mermean_zonint(
    data: np.ndarray | list[np.ndarray], lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    # mermean
    coslat = np.cos(lat / 180 * np.pi)[:, None]
    data = np.nanmean(data * coslat, axis=-2) / np.nanmean(coslat)

    # interpolation
    data = ct.interp_1d(lon, data, LON, axis=-1)
    return data


class PC:
    def __init__(self, pc1: np.ndarray, pc2: np.ndarray):
        self.pc1 = pc1
        self.pc2 = pc2


class DATA:
    def __init__(self, obs_pc: PC, mod_pcs: list[PC]):
        self.obs_pc = obs_pc
        self.mod_pcs = mod_pcs


def plot_phase_diagram(data: DATA, init_date: float | int) -> None:
    fig_name = tt.float2format(init_date, f"{FIG_DIR}/pd_%y%m%d.png")
    fig, ax = phase_diagram()

    # plot pcs
    ax.plot(data.obs_pc.pc1, data.obs_pc.pc2, label="obs", color="k")
    for ensname, mod_pc, color in zip(ENS_NAME, data.mod_pcs, COLORS):
        ax.plot(mod_pc.pc1, mod_pc.pc2, label=ensname, color=color)

    ax.plot(data.obs_pc.pc1[0], data.obs_pc.pc2[0], color="k", marker="o")

    # decorations
    ax.legend()
    ax.set_title(f"init_date:{tt.float2format(init_date)}")

    # save
    fig.savefig(fig_name)
    print(f"saved to {fig_name}")


def plot_curves(data: DATA, init_date: float | int) -> None:
    fig_name = tt.float2format(init_date, f"{FIG_DIR}/cu_%y%m%d.png")

    fig = plt.figure(layout="constrained")
    # plot pcs
    x = list(range(1, NUM_LEADS + 1))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(x, data.obs_pc.pc1, label="obs", color="k")
    for ensname, mod_pc, color in zip(ENS_NAME, data.mod_pcs, COLORS):
        ax1.plot(x, mod_pc.pc1, label=ensname, color=color)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(x, data.obs_pc.pc2, label="obs", color="k")
    for ensname, mod_pc, color in zip(ENS_NAME, data.mod_pcs, COLORS):
        ax2.plot(x, mod_pc.pc2, label=ensname, color=color)

    # decorations
    ax1.legend()
    ax1.set_title(f"init_date:{tt.float2format(init_date)}")

    # save
    fig.savefig(fig_name)
    print(f"saved to {fig_name}")


def read_data(init_date: float | int) -> DATA:
    def read_obs():
        obs_path = "../external/bom_rmm.nc"
        minmaxs = [[init_date + 1, init_date + NUM_LEADS]]
        obs_pc1, _ = nct.ncreadByDimRange(obs_path, "pc1", minmaxs)
        obs_pc2, _ = nct.ncreadByDimRange(obs_path, "pc2", minmaxs)
        return PC(obs_pc1, obs_pc2)

    def read_mod(ensname: str):
        mod_path = tt.float2format(
            init_date, f"./data/output/%Y/MJORMM_R60_%Y%m%d_{ensname}.txt"
        )

        pc1s, pc2s = [], []
        with open(mod_path, "rt") as f:
            f.readline()  # skip header
            for i in range(99):  # some safe number
                strline = f.readline()

                if not isinstance(strline, str) or not strline:
                    break

                numbers = [float(s) for s in strline.split()]
                pc1, pc2 = numbers[3:5]
                pc1s.append(pc1)
                pc2s.append(pc2)

        return PC(np.array(pc1s), np.array(pc2s))

    obs_pc = read_obs()
    mod_pcs = [read_mod(e) for e in ENS_NAME]
    return DATA(obs_pc, mod_pcs)


def phase_diagram():
    fig, ax = plt.subplots(figsize=(7, 7))
    ##
    angles = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(angles), np.sin(angles), color="k", linewidth=0.5)

    thisStyle = {"color": "k", "linewidth": 0.5, "linestyle": "--"}
    ax.plot([0, 0], [1, 4], **thisStyle)
    ax.plot([0, 0], [-1, -4], **thisStyle)
    ax.plot([1, 4], [0, 0], **thisStyle)
    ax.plot([-1, -4], [0, 0], **thisStyle)
    s2 = np.sqrt(2) / 2
    ax.plot([-s2, -4], [-s2, -4], **thisStyle)
    ax.plot([s2, 4], [-s2, -4], **thisStyle)
    ax.plot([-s2, -4], [s2, 4], **thisStyle)
    ax.plot([s2, 4], [s2, 4], **thisStyle)

    thisStyle = {"color": "k", "fontsize": 14}
    d1, d2 = 1.5, 3.5
    ax.text(-d2, -d1, str(1), **thisStyle)
    ax.text(-d1, -d2, str(2), **thisStyle)
    ax.text(d1, -d2, str(3), **thisStyle)
    ax.text(d2, -d1, str(4), **thisStyle)
    ax.text(d2, d1, str(5), **thisStyle)
    ax.text(d1, d2, str(6), **thisStyle)
    ax.text(-d1, d2, str(7), **thisStyle)
    ax.text(-d2, d1, str(8), **thisStyle)

    thisStyle = {
        "verticalalignment": "center",
        "horizontalalignment": "center",
        "fontsize": 14,
    }
    ax.text(0, -3.5, "Indian\nOcean", rotation=0, **thisStyle)
    ax.text(3.5, 0, "Maritime\nContinent", rotation=-90, **thisStyle)
    ax.text(0, 3.5, "Western\nPacific", rotation=0, **thisStyle)
    ax.text(-3.5, 0, "West. Hemi.\nand Africa", rotation=90, **thisStyle)

    ax.axis("equal")
    ax.set_xticks(np.r_[-4:4.5:1])
    ax.set_yticks(np.r_[-4:4.5:1])
    ax.set_xticks(np.r_[-4:4.5:0.5], minor=True)
    ax.set_yticks(np.r_[-4:4.5:0.5], minor=True)
    ax.set_xlim((-4.0, 4.0))
    ax.set_ylim((-4.0, 4.0))

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.tick_params(axis="both", which="major", length=8)
    ax.tick_params(axis="both", which="minor", length=4)
    return fig, ax


if __name__ == "__main__":
    main()
