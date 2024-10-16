
### Calculate the anomaly correlation coefficient (ACC) of model-analog (MA) forecasts 
# in comparison to both non-detrended GPCP and detrended GPCP precipitation 

import xarray as xr
import numpy as np
import pandas as pd

from functions_misc import get_obs_4pred_2D_pd, to_monthly, detrend_dim, get_da_allmodels_fromDSdict

# define list of model and ensemble names
Models_list = ["GFDL-ESM4",
           "HadGEM3-GC31-LL", 
           "HadGEM3-GC31-MM", 
           "NorESM2-LM", 
           "UKESM1-0-LL"]
Ensembles_list=["r1i1p1f1",
              "r1i1p1f1",
              "r1i1p1f1",
              "r1i1p1f1",
              "r1i1p1f2"]


# time period for ACC
ini_yr=1979
end_yr=2019

# define folder path name and output location path
ma_folder_path="/Projects/SeasonalFcst/data/other_analyses/SMYLE_Dillon/"
ma_output_path=(
    ma_folder_path+"OUTPUT/FOSIsst_30S30N/Work/"+
    "analogFCST/selA_2x2_newgrid/30N30S0E0W/MAFcst_2x2_newgrid/"
)

### =========================================
### ==== read Model Analog result (30S-30N)  ==== 
print("read model-analog output")
ds_precip_MA_30SN_dict={}
# loop through models
for ii, Modelname in enumerate(Models_list):
    print(Modelname)
    #get the fn_list(filename list) within the time period
    fn_list=[] 
    for YEAR in np.arange(ini_yr,end_yr+1):
        for month in np.arange(1,12+1):
            fn_list.append(
                ma_output_path+"CMIP6/"+Modelname+"/"+Ensembles_list[ii]+
                "/precip." +str(YEAR)+str(month).zfill(2)+".analog_fcst_ensembles.nc"
            )
    # read precip model analog output        
    ds_precip_MA_30SN_temp=xr.open_mfdataset(fn_list,combine='nested',concat_dim='time')
    # get the corresponding time dimension
    ds_precip_MA_30SN_temp=ds_precip_MA_30SN_temp.assign_coords(
        {'time':pd.date_range(start=str(ini_yr)+'-01-01', end=str(end_yr)+'-12-31', freq='MS')}
    )
    # save as dictionary
    ds_precip_MA_30SN_dict[Modelname+'_'+Ensembles_list[ii]]=ds_precip_MA_30SN_temp.transpose("time","lag",...) 
    
# clear memory
del ds_precip_MA_30SN_temp 

# define constant for changing precip unit 
scale = 86400. #1 kg/m2/s = 86400 mm/day

### ====================
### ==== read GPCP  ====
print('read GPCP')
fn_list=[]
# list of the GPCP data we want to read (add 2 more years to include the verification time for 2 year (24 mon) forecast)
for YEAR in np.arange(ini_yr,end_yr+2+1):
    fn_list.append("/Projects/LIM/Realtime/Realtime/Anom/precip.GPCP.2x2.anom.1979-2019."+str(YEAR)+".nc")

# read GPCP precip             
ds_GPCP=xr.open_mfdataset(fn_list,combine='nested',concat_dim='time')
# get the corresponding time dimension
ds_GPCP=ds_GPCP.assign_coords({'time':pd.date_range(start=str(ini_yr)+'-01-01', end=str(end_yr+2)+'-12-31', freq='MS')})


# calculate detrend GPCP precip
da_GPCP_detrend=detrend_dim(ds_GPCP.precip,'time',1).compute()

# reshape the GPCP data into the same as model-analog forecast
da_GPCP_4prediction=get_obs_4pred_2D_pd(ds_GPCP.precip,24,ini_yr,end_yr)
da_GPCPdetrend_4prediction=get_obs_4pred_2D_pd(da_GPCP_detrend,24,ini_yr,end_yr)

# cal GPCP seasonal mean
da_GPCP_4prediction_seaM=da_GPCP_4prediction.rolling(lag=3,center=False).mean().shift(lag=-2).dropna(dim='lag').compute()
da_GPCPdetrend_4prediction_seaM=da_GPCPdetrend_4prediction.rolling(lag=3,center=False).mean().shift(lag=-2).dropna(dim='lag').compute()



# change the dictionary to DataArray 
da_precip_MA_30SN_allmodel=get_da_allmodels_fromDSdict(ds_precip_MA_30SN_dict,Models_list,Ensembles_list,'precip')
del ds_precip_MA_30SN_dict

## multiply the scale for precip (change units from kg/m2/s to mm/day)
da_precip_MA_30SN_allmodel=da_precip_MA_30SN_allmodel*scale




### ==== calculate and save ACC for each model ====
# number of ensemble members of model-analog
nens=20
# loop though models
for ii, Modelname in enumerate(Models_list):  
    print(Modelname)
    
    # calculate enM for one model  
    da_precip_MA_30SN_OneModelensM=da_precip_MA_30SN_allmodel.sel(model=Modelname+'_'+Ensembles_list[ii]).sel(ensemble=slice(0,nens-1)).mean(dim='ensemble').compute()

    # calculate seasonal mean for one model
    print('cal MA seasonal mean')
    da_precip_MA_30SN_OneModelensM_seaM=da_precip_MA_30SN_OneModelensM.rolling(lag=3, center=False).mean().shift(lag=-2).dropna(dim='lag').compute()
    del da_precip_MA_30SN_OneModelensM

    # change to monthly 
    da_precip_MA_30SN_OneModelensM_seaM_monthly=to_monthly(da_precip_MA_30SN_OneModelensM_seaM)
    del da_precip_MA_30SN_OneModelensM_seaM

    # calculate ACC for nodetrend and detrend
    print('cal ACC')
    da_precip_corr_30SN_ensM_OneModel_GPCPnodetrend_seaM=xr.corr(da_GPCP_4prediction_seaM,da_precip_MA_30SN_OneModelensM_seaM_monthly,dim='year').compute()
    da_precip_corr_30SN_ensM_OneModel_GPCPdetrend_seaM=xr.corr(da_GPCPdetrend_4prediction_seaM,da_precip_MA_30SN_OneModelensM_seaM_monthly,dim='year').compute()
    del da_precip_MA_30SN_OneModelensM_seaM_monthly

    print('save seaM for one model')
    # save nodetrend ACC
    ds_precip_corr_30SN_ensM_OneModel_GPCPnodetrend_seaM=xr.Dataset()
    ds_precip_corr_30SN_ensM_OneModel_GPCPnodetrend_seaM['precip']=da_precip_corr_30SN_ensM_OneModel_GPCPnodetrend_seaM
    ds_precip_corr_30SN_ensM_OneModel_GPCPnodetrend_seaM.to_netcdf(
        ma_folder_path+'analyses/ACC/test_shared/ds_precip_corr_30SN_ensM_'+Modelname+'_'+Ensembles_list[ii]+'_GPCPnodetrend_seaM.nc'
    )

    # save detrend ACC
    ds_precip_corr_30SN_ensM_OneModel_GPCPdetrend_seaM=xr.Dataset()
    ds_precip_corr_30SN_ensM_OneModel_GPCPdetrend_seaM['precip']=da_precip_corr_30SN_ensM_OneModel_GPCPdetrend_seaM
    ds_precip_corr_30SN_ensM_OneModel_GPCPdetrend_seaM.to_netcdf(
        ma_folder_path+'analyses/ACC/test_shared/ds_precip_corr_30SN_ensM_'+Modelname+'_'+Ensembles_list[ii]+'_GPCPdetrend_seaM.nc'
    )

