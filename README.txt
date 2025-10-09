The source files (both clim data and code) are located on ln23,
then copied to ln16 for operational running.


---- ---- run ---- ----
1. create a directory for running (let's call it RUNDIR)
2. copy "0_setup.sh" into RUNDIR and execute it
3. try "run.sh > run.out"
4. check the on-screen output or ./logs for errors
5. install to crontab: "00 08 * * * cd $RUNDIR && sh run.sh > run.out"


---- ---- logic ---- ----
output is created everyday
    variables: RMM1, RMM2 (derived from u850, olr, u200)
    area: 15S-15N, 0-360
    forecast: 40 days, T+1 to T+40 (T = reference date. Square braketed times denote init dates.)
    analysis: 120 days, T-119 to T

forecasts (three members from three init time steps):
    valid for T+1, T+2, ..., T+40, | backup plan ("degraded" outputs) for missing files:
    lead =    2~41, 3~42, 4~43     |   lead = 5~44,  6~45
    init from [T-1], [T-2], [T-3], |   init = [T-4], [T-5]

analysis:
    valid for T-119, T-118, ..., T-1, T-0         | backup plan for missing files: 
    lead = 1, 1,            ..., 1,   2           |   (1) using the neighboring dates to average
    init from [T-120], [T-119], ..., [T-2], [T-2] |   (2) using lead=2
                                                      i.e, fail if 3 continuous files are missing

degraded output:
    The output is degraded when switched to the backup plan. A warning message will be
    created in the run directory. The warning message will be removed if a new output 
    is updated by the expected normal procedures (left hand side of the above tow paragraphs).

steps:
    source (grib2 or tarred grib2)
1_convertOp2nc.py
    It cuts out the needed data (u200, u850, OLR) from source files to nc.
    The date will be auto skipped if the source file does not exist,
                                     or the middle nc file already exists.
2_nc2ascii.py
    It aligns the forecasts and analysis, corrects the model climate bias,
    calculates RMM indices, and writes to the output ASCII format.

(verify.py)
    check: draw the output and compare to the BOM data


---- ---- CFSv2 operation timing ---- ----
    e.g., for init: 2025/03/09 00Z
    daymean output finished in 2025/03/09 20:00L ~ 2025/03/10 10:00L 
    (but the nearest 40 day forecast is earlier, so run at 08:00L should be ok)
    -> latest output varilable = [T-1]


---- ---- output format ---- ----
./data/output/MJORMM_R60_$yyyy$mm$dd_m{000,001,002,avg}.txt
    m00[0-2] - lag ensemble member 0-2
    mavg     - lag ensemble average
                 ...

---- ---- paths ---- ----
model data
    on ln16/17:
        - /nwpr/cfsop/cfsaoper/P6/OP/WORKING/tcoTL359l60m550x50oocb4_rsmwrk
        - /cfsdata/cfsaoper/P6/OP/CWBCFSv2/gsmdm/gsmdm_source
    on silo: /op/arc/cfm/P6/CWBCFSv2
    on sata1 (accessible from rdccs1): /syn_sata1/users/cfsoper/P6/CWBCFSv2/gsmdm/gsmdm_source

model clim data
    on ln23: /nwpr/gfs/com120/9_data/models/processed/re_cfsv2_dm/clim

BOM data for comparison (verify.py)
    on ln23: /nwpr/gfs/com120/5_CFS_MJO/external
    accessed from BOM website

code: on ln23 /nwpr/gfs/com120/5_CFS_MJO/source
