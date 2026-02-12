"""
postblock.py
-------------------------------------------------------
Content:
    - PostBlock
    - TracerFixer
"""

import torch
from torch import nn

import numpy as np

from credit.data import get_forward_data
from credit.transforms import load_transforms
import logging

logger = logging.getLogger(__name__)


class PostBlock(nn.Module):
    def __init__(self, post_conf):
        """
        post_conf: dictionary with config options for PostBlock.
                   if post_conf is not specified in config,
                   defaults are set in the parser

        This class is a wrapper for all post-model operations.
        Registered modules:
            - SKEBS
            - TracerFixer
            - GlobalMassFixer
            - GlobalEnergyFixer

        """
        super().__init__()

        self.operations = nn.ModuleList()

        # The general order of postblock processes:
        # (1) tracer fixer --> mass fixer --> SKEB / water fixer --> energy fixer

        # negative tracer fixer
        if post_conf["tracer_fixer"]["activate"]:
            logger.info("TracerFixer registered")
            opt = TracerFixer(post_conf)
            self.operations.append(opt)

        # stochastic kinetic energy backscattering (SKEB)
        if post_conf["skebs"]["activate"]:
            logging.info("using SKEBS")
            self.operations.append(SKEBS(post_conf))

        # global mass fixer
        if post_conf["global_mass_fixer"]["activate"]:
            if post_conf["global_mass_fixer"]["activate_outside_model"] is False:
                logger.info("GlobalMassFixer registered")
                opt = GlobalMassFixer(post_conf)
                self.operations.append(opt)

        # global water fixer
        if post_conf["global_water_fixer"]["activate"]:
            if post_conf["global_water_fixer"]["activate_outside_model"] is False:
                logger.info("GlobalWaterFixer registered")
                opt = GlobalWaterFixer(post_conf)
                self.operations.append(opt)

        # global energy fixer
        if post_conf["global_energy_fixer"]["activate"]:
            if post_conf["global_energy_fixer"]["activate_outside_model"] is False:
                logger.info("GlobalEnergyFixer registered")
                opt = GlobalEnergyFixer(post_conf)
                self.operations.append(opt)

    def forward(self, x):
        for op in self.operations:
            x = op(x)

        if isinstance(x, dict):
            # if output is a dict, return y_pred (if it exists), otherwise return x
            return x.get("y_pred", x)
        else:
            # if output is not a dict (assuming tensor), return x
            return x


# class PrecipClimateGuide(nn.Module):
#     def __init__(self, post_conf):
#         super().__init__()
        
#         # ------------------------------------------------------------------------------------ #
#         # initialize physics computation

#         # provide example data if it is a unit test
#         if post_conf["precip_climate_guide"]["simple_demo"]:
#             pass
            
#         else:
#             self.flag_mean = post_conf["precip_climate_guide"]["mean_adjust"]
            
#             # get climatology mean and std as reference
#             ds_clim = get_forward_data(post_conf["data"]["save_loc_climatology"])
            
#             if self.flag_mean:
#                 varname_precip_mean = post_conf["precip_climate_guide"]["varname_precip_mean"]
#                 self.precip_mean = ds_clim[varname_precip_mean].values.astype(np.float32)
            
#             varname_precip_std = post_conf["precip_climate_guide"]["varname_precip_std"]
#             self.precip_std = ds_clim[varname_precip_std].values.astype(np.float32)
            
#             # get mon-of-year and day-of-mon
#             dates = ds_clim['time'].values
#             ds_clim.close() # close
#             self.ref_mon = (dates.astype('datetime64[M]').astype('int64') % 12) + 1
#             self.ref_day = (dates.astype('datetime64[D]') - dates.astype('datetime64[M]')).astype(int) + 1
#             self.ref_hour = dates.astype('datetime64[h]').astype('int64') % 24
            
#             # the weights of forecasted mean and std
#             # (1-weights) is the weights of climatology
#             max_lead_time = post_conf["precip_climate_guide"]["max_lead_time"]
#             weights_type = post_conf["precip_climate_guide"]["weights"]
            
#             # exponential decay weights
#             if weights_type == 'exp':
#                 n = np.linspace(0, 6*max_lead_time-1, 6*max_lead_time)
#                 k = (2 - np.log(2)) / (max_lead_time - 1)
#                 self.weights = (0.5 * np.exp(-k * n)).astype(np.float32)
#             elif isinstance(weights_type, (int, float)):
#                 # linear weights
#                 self.weights = weights_type*np.ones(6*max_lead_time).astype(np.float32)
#             else:
#                 self.weights = 0.01*np.ones(6*max_lead_time).astype(np.float32)
                
#             self.weights_len = len(self.weights)
#             # -------------------------------------------------------------------------- #

#         # ------------------------------------------------------------------------------------ #
#         # identify variables of interest
#         self.precip_ind = int(post_conf["precip_climate_guide"]["precip_ind"])
        
#         # ------------------------------------------------------------------------------------ #
#         # setup a scaler
#         if post_conf["precip_climate_guide"]["denorm"]:
#             self.state_trans = load_transforms(post_conf, scaler_only=True)
#         else:
#             self.state_trans = None

#     def forward(self, x, dt_nanoseconds, forecast_step):
#         # ------------------------------------------------------------------------------ #
#         # get tensors
#         # x_input (batch, var, time, lat, lon)
#         y_pred = x["y_pred"]
        
#         # other needed inputs
#         N_vars = y_pred.shape[1]
        
#         # if denorm is needed
#         if self.state_trans:
#             y_pred = self.state_trans.inverse_transform(y_pred)

#         precip = y_pred[:, self.precip_ind, 0, ...]

#         # --------------------------------------------------- #
#         # get weights
#         if forecast_step - 1 < self.weights_len: 
#             weights = self.weights[forecast_step-1]
#         else:
#             weights = self.weights[-1]

#         # get the input batch size
#         batch_size = len(y_pred)
#         # loop through all samples in a batch
#         for i_batch in range(batch_size):
#             # get climate ref
#             dt_nanosecond = int(dt_nanoseconds[i_batch]) # <-- tensor to int
#             dt_fcst = np.datetime64(dt_nanosecond, 'ns')
#             mon_fcst = (dt_fcst.astype('datetime64[M]').astype('int64') % 12) + 1
#             day_fcst = (dt_fcst.astype('datetime64[D]') - dt_fcst.astype('datetime64[M]')).astype(int) + 1
#             hour_fcst = dt_fcst.astype('datetime64[h]').astype('int64') % 24

#             match_mask = (self.ref_mon == mon_fcst) & (self.ref_day == day_fcst) & (self.ref_hour == hour_fcst)
#             ref_index = np.nonzero(match_mask)[0][0]
            
#             precip_sample = precip[i_batch, ...]
            
#             # mean value
#             precip_mean = np.float32(precip_sample.mean().item())
            
#             if self.flag_mean:
#                 ref_mean = self.precip_mean[ref_index]    
#                 correct_mean = weights*precip_mean + (1-weights)*ref_mean
#             else:
#                 correct_mean = precip_mean

#             # std value
#             ref_std = self.precip_std[ref_index]
#             precip_std = np.float32(precip_sample.std().item())
#             correct_std = weights*precip_std + (1-weights)*ref_std

#             # correction
#             precip_sample = (precip_sample - precip_mean) * (correct_std/precip_std) + precip_mean
#             precip_sample[precip_sample<0] = 0.0
#             precip_mean_new = np.float32(precip_sample.mean().item())
            
#             precip_sample = precip_sample * (correct_mean/precip_mean_new)
            
#             precip[i_batch, ...] = precip_sample
            
#         # ===================================================================== #
#         # return fixed precip back to y_pred
#         precip = precip.unsqueeze(1).unsqueeze(2)
#         y_pred = concat_fix(y_pred, precip, self.precip_ind, self.precip_ind, N_vars)

#         if self.state_trans:
#             y_pred = self.state_trans.transform_array(y_pred)
            
#         # give it back to x
#         x["y_pred"] = y_pred

#         # return dict, 'x' is not touched
#         return x

class TracerFixer(nn.Module):
    """
    This module fixes tracer values by replacing their values to a given threshold
    (e.g., `tracer[tracer<thres] = thres`).

    Args:
        post_conf (dict): config dictionary that includes all specs for the tracer fixer.
    """

    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------ #
        # identify variables of interest
        self.tracer_indices = post_conf["tracer_fixer"]["tracer_inds"]
        self.tracer_thres = post_conf["tracer_fixer"]["tracer_thres"]

        # ------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf["tracer_fixer"]["denorm"]:
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None

    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get y_pred
        # y_pred is channel first: (batch, var, time, lat, lon)
        y_pred = x["y_pred"]

        # if denorm is needed
        if self.state_trans:
            y_pred = self.state_trans.inverse_transform(y_pred)

        # ------------------------------------------------------------------------------ #
        # tracer correction
        for i, i_var in enumerate(self.tracer_indices):
            # get the tracers
            tracer_vals = y_pred[:, i_var, ...]

            # in-place modification of y_pred
            thres = self.tracer_thres[i]
            tracer_vals[tracer_vals < thres] = thres

        if self.state_trans:
            y_pred = self.state_trans.transform_array(y_pred)

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x
