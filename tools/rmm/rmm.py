from .. import nctools as nct
import numpy as np

# read in the static data for calculating the pcs
var_names: list[str] = ['olr', 'u200', 'u850']
static_data_path: str = '/nwpr/gfs/com120/tools/pytools/rmm/static_rmm.nc'
statics: dict[str: dict[str: np.ndarray]] = {
    vn: {
        'eof': nct.read(static_data_path, f'eof_{vn}'),
        'mean': nct.read(static_data_path, f'mean_{vn}'),
        'variance': nct.ncreadatt(static_data_path, '/', f'var_{vn}'),
    }
    for vn in var_names
}
statics['std_pc1'] = nct.ncreadatt(static_data_path, '/', f'std_pc1')
statics['std_pc2'] = nct.ncreadatt(static_data_path, '/', f'std_pc2')


def cal_pc(
    olr:np.ndarray, 
    u200:np.ndarray,
    u850:np.ndarray, 
    sub120:bool=True
) -> tuple[np.ndarray, np.ndarray]:

    data = {'olr': olr, 'u200': u200, 'u850': u850}

    # normalize the data
    for vn in var_names:
        data[vn] = (data[vn] - statics[vn]['mean']) / np.sqrt(statics[vn]['variance'])

    if sub120:
        data = _remove_previous_120(data)

    # projection along the longitude
    pc1 = [
        np.nansum(data[vn] * statics[vn]['eof'][0, :], axis=-1)
        for vn in var_names
    ]

    pc2 = [
        np.nansum(data[vn] * statics[vn]['eof'][1, :], axis=-1)
        for vn in var_names
    ]

    # sum over the three variables and divide by the std(pc)
    pc1 = np.nansum(pc1, axis=0) / statics['std_pc1']
    pc2 = np.nansum(pc2, axis=0) / statics['std_pc2']

    return pc1, pc2


def cal_phase(pc1, pc2):
    angleDeg = np.angle(pc1 + 1j * pc2) / np.pi * 180
    phase = np.nan * np.ones_like(pc1)

    phase[( 0<= angleDeg) & (angleDeg < 45)] = 5
    phase[( 45<= angleDeg) & (angleDeg < 90)] = 6
    phase[( 90<= angleDeg) & (angleDeg < 135)] = 7
    phase[( 135<= angleDeg) & (angleDeg <= 180)] = 8
    phase[( -180<= angleDeg) & (angleDeg < -135)] = 1
    phase[( -135<= angleDeg) & (angleDeg < -90)] = 2
    phase[( -90<= angleDeg) & (angleDeg < -45)] = 3
    phase[( -45<= angleDeg) & (angleDeg < 0)] = 4
    return phase


def _remove_previous_120(data:dict[str:np.ndarray]) -> dict[str:np.ndarray]:
    return {
        vn: remove_previous_runmean(data[vn], n=120)
        for vn in var_names
    }


def remove_previous_runmean(y0:np.ndarray, n:int):
    nt = y0.shape[0]
    y1 = np.nan * np.ones(y0.shape)

    for it in range(nt):
        if it < n:
            ts = 0
        else:
            ts = it - n

        y1[it, :] = y0[it, :] - np.nanmean(y0[ts:it, :], axis=0)

    return y1
