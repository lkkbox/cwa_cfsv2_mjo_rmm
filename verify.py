#!/usr/bin/env python
from tools import timetools as tt
from tools import nctools as nct
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Agg")


NUM_LEADS = 40
ENS_NAME = ['m000', 'm001', 'm002', 'mavg']
COLORS = [
    (0.5, 0, 0),
    (1, 0, 0),
    (1, 0.5, 0.5),
    (0, 0.8, 0.5),
]
FIG_DIR = './figs'


class PC:
    def __init__(self, pc1:np.ndarray, pc2:np.ndarray):
        self.pc1 = pc1
        self.pc2 = pc2

class DATA:
    def __init__(self, obs_pc:PC, mod_pcs:list[PC]):
        self.obs_pc = obs_pc
        self.mod_pcs = mod_pcs


def main():
    init_date = tt.ymd2float(2025, 8, 20)
    data = read_data(init_date)
    plot_phase_diagram(data, init_date)
    plot_curves(data, init_date)


def plot_phase_diagram(data: DATA, init_date:float|int) -> None:
    fig_name = tt.float2format(init_date, f'{FIG_DIR}/pd_%y%m%d.png')
    fig, ax = phase_diagram()

    # plot pcs
    ax.plot(data.obs_pc.pc1, data.obs_pc.pc2, label='obs', color='k')
    for ensname, mod_pc, color in zip(ENS_NAME, data.mod_pcs, COLORS):
        ax.plot(mod_pc.pc1, mod_pc.pc2, label=ensname, color=color)

    ax.plot(data.obs_pc.pc1[0], data.obs_pc.pc2[0], color='k', marker='o')

    # decorations
    ax.legend()
    ax.set_title(f'init_date:{tt.float2format(init_date)}')

    # save
    fig.savefig(fig_name)
    print(f'saved to {fig_name}')


def plot_curves(data: DATA, init_date:float|int) -> None:
    fig_name = tt.float2format(init_date, f'{FIG_DIR}/cu_%y%m%d.png')

    fig = plt.figure(layout='constrained')
    # plot pcs
    x = list(range(1, NUM_LEADS+1))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, data.obs_pc.pc1, label='obs', color='k')
    for ensname, mod_pc, color in zip(ENS_NAME, data.mod_pcs, COLORS):
        ax1.plot(x, mod_pc.pc1, label=ensname, color=color)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, data.obs_pc.pc2, label='obs', color='k')
    for ensname, mod_pc, color in zip(ENS_NAME, data.mod_pcs, COLORS):
        ax2.plot(x, mod_pc.pc2, label=ensname, color=color)

    # decorations
    ax1.legend()
    ax1.set_title(f'init_date:{tt.float2format(init_date)}')

    # save
    fig.savefig(fig_name)
    print(f'saved to {fig_name}')




def read_data(init_date:float|int) -> DATA:
    def read_obs():
        obs_path = '../external/bom_rmm.nc'
        minmaxs = [[init_date+1, init_date+NUM_LEADS]]
        obs_pc1, _ = nct.ncreadByDimRange(obs_path, 'pc1', minmaxs)
        obs_pc2, _ = nct.ncreadByDimRange(obs_path, 'pc2', minmaxs)
        return PC(obs_pc1, obs_pc2)

    def read_mod(ensname:str):
        mod_path = tt.float2format(
            init_date,
            f'./data/output/%Y/MJORMM_R60_%Y%m%d_{ensname}.txt'
        )

        pc1s, pc2s = [], []
        with open(mod_path, 'rt') as f:
            f.readline() # skip header
            for i in range(99): # some safe number
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
    angles = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(angles), np.sin(angles), color='k', linewidth=0.5)

    thisStyle = {'color': 'k', 'linewidth': 0.5, 'linestyle': '--'}
    ax.plot([0,  0], [1, 4], **thisStyle)
    ax.plot([0,  0], [-1, -4], **thisStyle)
    ax.plot([1,  4], [0, 0], **thisStyle)
    ax.plot([-1, -4], [0, 0], **thisStyle)
    s2 = np.sqrt(2)/2
    ax.plot([-s2,  -4], [-s2, -4], **thisStyle)
    ax.plot([s2,   4], [-s2, -4], **thisStyle)
    ax.plot([-s2,  -4], [s2,  4], **thisStyle)
    ax.plot([s2,   4], [s2,  4], **thisStyle)

    thisStyle = {'color': 'k', 'fontsize': 14}
    d1, d2 = 1.5, 3.5
    ax.text(-d2, -d1, str(1), **thisStyle)
    ax.text(-d1, -d2, str(2), **thisStyle)
    ax.text(d1, -d2, str(3), **thisStyle)
    ax.text(d2, -d1, str(4), **thisStyle)
    ax.text(d2,  d1, str(5), **thisStyle)
    ax.text(d1,  d2, str(6), **thisStyle)
    ax.text(-d1,  d2, str(7), **thisStyle)
    ax.text(-d2,  d1, str(8), **thisStyle)

    thisStyle = {
        'verticalalignment': 'center',
        'horizontalalignment': 'center',
        'fontsize': 14
    }
    ax.text(
        0, -3.5, 'Indian\nOcean', rotation=0, **thisStyle
    )
    ax.text(
        3.5, 0, 'Maritime\nContinent', rotation=-90, **thisStyle
    )
    ax.text(
        0, 3.5, 'Western\nPacific', rotation=0, **thisStyle
    )
    ax.text(
        -3.5, 0, 'West. Hemi.\nand Africa', rotation=90,  **thisStyle
    )

    ax.axis('equal')
    ax.set_xticks(np.r_[-4:4.5:1])
    ax.set_yticks(np.r_[-4:4.5:1])
    ax.set_xticks(np.r_[-4:4.5:0.5], minor=True)
    ax.set_yticks(np.r_[-4:4.5:0.5], minor=True)
    ax.set_xlim((-4.0, 4.0))
    ax.set_ylim((-4.0, 4.0))

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    ax.tick_params(axis='both', which='major', length=8)
    ax.tick_params(axis='both', which='minor', length=4)
    return fig, ax


if __name__ == '__main__':
    main()
