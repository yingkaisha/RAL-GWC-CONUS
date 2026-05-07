# `TEST`: debugging configs

* Single-step pre-training (multi-GPU): `model_single.yml`, `launch_single.sh`
* Single-step pre-training (single-GPU): `model_single_none.yml`
* Multi-step fine-tuning: `model_multi_01.yml`, `launch_multi_01.sh`
* Inference tests: `model_predict_clim.yml`, `launch_predict_clim.sh`

# `CONUS_GP_clean`: model configs

* Single-step pre-training: `model_single.yml`, `launch_single.sh`
* Multi-step fine-tuning: `model_multi_01.yml`, `launch_multi_01.sh`, ..., `model_multi_05.yml`, `launch_multi_05.sh`
* Model inference:
  - Downscale 3-hourly ERA5: `model_clim_B3H.yml`, `launch_clim_B3H.sh`
  - Downscale 6-hourly ERA5: `model_clim_B6H.yml`, `launch_clim_B6H.sh`
  - Downscale GDAS/FNL: `model_clim_GDAS.sh`, `launch_clim_GDAS.sh`
  - Downscale CESM-LENS2 30-year 1980-2010 historical runs: `model_clim_CESM_HIST.yml`, `launch_clim_CESM_HIST.sh`
  - Downscale CESM-LENS2 30-year 2070-2100 SSP370 runs: `model_clim_CESM_SSP.yml`, `launch_clim_CESM_SSP.sh`
  - Downscale CESM-LENS2 MOAR members: `model_clim_CESM_mem00.yml`, `launch_clim_CESM_mem00.sh`, ..., `model_clim_CESM_mem09.yml`, `launch_clim_CESM_mem09.sh`

  *Others: `_p2` stands for "part2"; they break long inference runs into multiple PBS jobs.

# `CONUS_GP_diag_unet`: U-net post-processing model configs

* Training: `model_single.yml`, `launch_single.sh`
* Model inference:
    - 3-hourly ERA5: `model_opt_B3H.yml`, `launch_B3H.sh`
    - 3-hourly ERA5: `model_opt_B3H.yml`, `launch_B3H.sh`
    - GDAS/FNL: `model_opt_GDAS.yml`, `launch_GDAS.sh`