# AI-based Limited Area Model for Dynamical Downscaling of Long-term Kilometer-scale Weather Information

Yingkai Sha, Tracy Hertneky, Ethan Gutmann, Seth McGinnis, Lulin Xue, David John Gagne II, Kathryn Newman, Andrew Newman

NSF National Center for Atmospheric Research, Boulder, Colorado, USA

## Abstract

An AI-based Limited-Area Model (LAM) is developed for dynamical downscaling over the Southern Great Plains and the southeastern United States, with strong generalization abilities under diverse boundary conditions. The model is trained using 0.25$^\circ$, 3-hourly ERA5 as forcings and CONUS404 as targets in 1980--2019, producing 4-km, hourly dynamical downscaling outputs; it is also connected to a post-processing model to derive additional diagnostic variables. The model is evaluated across multiple forcing datasets, time periods, and climate regimes. For present-day downscaling in the 2021--2024 water years, the model produces stable multi-year simulations with no unrealistic drift; its deterministic verification skills are comparable to other weather-forecasting-oriented AI models. The model also generalizes robustly to a 1.0$^\circ$, 6-hourly non-ERA5 forcing dataset, yielding only minor performance change. Frontal cyclone and hurricane case studies further demonstrate that the model reconstructs realistic, interpretable weather-scale dynamical and thermodynamic structure from coarse boundary information. The AI-based LAM is further tested by downscaling 30-year global climate model runs in 1980--2010 and 2070--2100, and climate model ensembles in 2025-2027. In this application, the model remains stable at hourly downscaling frequencies for all 30 years and effectively captures future climate-change signals, indicating meaningful generalization across different climate regimes. When downscaling ensembles, the model produces well-posed ensemble distributions without collapsing the ensemble spread. Overall, the AI-based LAM of this study offers good downscaling performance and generalization abilities. It provides a practical and transferable example of adapting AI weather prediction models for regional climate applications.

## Introduction

This repository contains code for data preprocessing, AI model training, inference, and result evaluation for the main paper.

The AI model training and inference part is modified from the [MILES-CREDIT](https://github.com/NCAR/miles-credit) platform. The object-oriented verification was conducted directly using the [METplus](https://dtcenter.org/software-tools/metplus) tool.

## Navigation

### Figures
* Downscaling domain information [[Link]](https://github.com/yingkaisha/RAL-GWC-CONUS/blob/main/_PAPER/visualization/FIG_Geoinfo.ipynb)
* RMSE and energy spectrum verification [[Link]](https://github.com/yingkaisha/RAL-GWC-CONUS/blob/main/_PAPER/visualization/FIG_RMSE_ZES.ipynb)
* Climatology comparison results [[Link]](https://github.com/yingkaisha/RAL-GWC-CONUS/blob/main/_PAPER/visualization/FIG_CLIM_compare.ipynb)
* Downscaled TMAX, TMIN verification [[Link]](https://github.com/yingkaisha/RAL-GWC-CONUS/blob/main/_PAPER/visualization/FIG_TMAX_TMIN.ipynb)
* Hurricane example [[Link]](https://github.com/yingkaisha/RAL-GWC-CONUS/blob/main/_PAPER/visualization/FIG_example_TC_case.ipynb) Frontal-cyclone example [[Link]](https://github.com/yingkaisha/RAL-GWC-CONUS/blob/main/_PAPER/visualization/FIG_example_Front_case.ipynb)
* CESM-LENS2 downscaling evaluations [[Link]](https://github.com/yingkaisha/RAL-GWC-CONUS/blob/main/_PAPER/visualization/FIG_CESM_30Y_T2.ipynb) [[Link]](https://github.com/yingkaisha/RAL-GWC-CONUS/blob/main/_PAPER/visualization/FIG_CESM_30Y_PWAT.ipynb) [[Link]](https://github.com/yingkaisha/RAL-GWC-CONUS/blob/main/_PAPER/visualization/FIG_CESM_ENS.ipynb)  

## Model Weights

coming soon

## Resources

coming soon

# Acknowledgement
This repository is based upon work supported by the National Science Foundation (NSF) National Center for Atmospheric Research (NCAR), which is a major facility sponsored by the U.S. National Science Foundation under Cooperative Agreement No. 1852977. 
Y. Sha and A. Newman are also supported by the Department of Defense (DoD) Environmental Security Technology Certification Program award \#W912HQ24C0083 (project NH24-8403). S. McGinnis is also supported by the Department of Energy RGCM program award DOE DE-SC0016605. We would like to acknowledge high-performance computing support from Derecho and Casper provided by the Computational and Information Systems Laboratory, NCAR, and sponsored by the NSF.
