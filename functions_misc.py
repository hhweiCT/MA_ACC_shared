import xarray as xr
import numpy as np
import pandas as pd

def get_obs_4pred_2D_pd(da_VAR_OBS,length_fc_month,start_yr,end_yr):
    """Reshape obs data to the structure of forecast data (year, month, lag ,lat, lon)

    year : initial year of the forecast
    month : initial month of the forecast
    lag: lead time of the forecast

    da_VAR_OBS : observation data (time series)
    length_fc_month : length of forecast month
    start_yr : start initial year of the forecast
    end_yr : end initial year of the forecast
    """
    
    ver_nyr=end_yr-start_yr+1

    da_VAR_OBS_4prediction=xr.DataArray(
                     data=np.zeros((ver_nyr,12,length_fc_month+1,np.size(da_VAR_OBS.lat),np.size(da_VAR_OBS.lon)))*np.nan,
                     dims=["year","month", "lag","lat","lon"],
                     coords=dict(
                         year=(["year"], np.arange(start_yr,end_yr+1)),
                         month=(["month"], np.arange(1,12+1)),
                         lag=(["lag"], np.arange(0,length_fc_month+1)),
                         lat=(["lat"], da_VAR_OBS.lat.data),
                         lon=(["lon"], da_VAR_OBS.lon.data)))

    # loop through 12 initial months of the forecast
    for mm in np.arange(0,12):
        # loop through ver_nyr initial years of the forecast
        # get the corresponding starting and ending verification datetime for each forecast
        for ver_yrr in np.arange(0,ver_nyr):
            # start of verification datetime
            specific_pd_datetime  = pd.datetime(
                da_VAR_OBS_4prediction.year[ver_yrr].values, da_VAR_OBS_4prediction.month[mm].values, 1, 0, 0, 0
            )
            # end of verification datetime
            specific_pd_datetime2 = specific_pd_datetime + pd.DateOffset(months=length_fc_month)

            # check the end time not exceeding the time period of the data we read
            if (specific_pd_datetime2 <= pd.datetime(end_yr+2, 12, 1, 0, 0, 0)):
                da_VAR_OBS_4prediction[ver_yrr,mm,:,:,:]=da_VAR_OBS.sel(
                    time=slice(specific_pd_datetime,specific_pd_datetime2)
                ).values                
            else:
                print(mm,ver_yrr)
                print(specific_pd_datetime2)
    return da_VAR_OBS_4prediction

def to_monthly(ds):
    """Reshape (time) dimension to (year, month)"""
    # assign year and month coordinates
    ds = ds.assign_coords(year=('time', ds.time.dt.year.values),
                          month=('time', ds.time.dt.month.values))
    # unstack time into year and month
    ds_monthly=ds.set_index(time=('year', 'month')).unstack('time').transpose("year","month",...)
    return ds_monthly


def detrend_dim(da, dim, deg=1):
    """detrend the data along a dimension)"""
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


def get_da_allmodels_fromDSdict(ds_box_dict,Model_list,Ensemble_list,varname):
    """Convert dictionary (Dataset) to DataArray"""

    # Convert the dictionary to a list of DataArrays
    da_box_list = [xr.DataArray(data[varname], dims=data[varname].dims, coords=data[varname].coords, name=name) for name, data in ds_box_dict.items()]
 
    # concat the list to get a new dimension (model)
    da_box_allmodels=xr.concat(da_box_list,dim='model')
    da_box_allmodels["model"]=[Modelname+"_"+Ensemble_list[ii] for ii, Modelname in enumerate(Model_list) ]
  
    return da_box_allmodels