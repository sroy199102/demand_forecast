# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 23:26:29 2021

@author: roysoumy
"""

import numpy as np
import pandas as pd
import re
import datetime as dt
from csv import reader
import statsmodels.api as sm
from sklearn import linear_model


from fbprophet import Prophet

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sqlalchemy import create_engine

import statsmodels.api as sm
import matplotlib

from pylab import rcParams
from matplotlib import pyplot as plt
from matplotlib import style
import itertools

import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

#map_file_loc = 'C:/Users/roysoumy/Documents/Ad-hoc/Current_map_20201224.csv'
#ddu_fcst_data_loc = "C:/Users/roysoumy/Documents/Demand Forecast Model/Results/Adhoc_"
#dow_days = 112
#weeks_ddu_fcst = 12
#launch_date = '2021-01-01'
#Publish = 'No'
#scale_ddu = "Yes"
#ddu_goals_loc = "C:/Users/roysoumy/Documents/Demand Forecast Model/ddu_2020.csv"
#caps_attainment_loc = "C:/Users/roysoumy/Documents/Demand Forecast Model/attn_factor_data.csv"
#
#in_loc = map_file_loc
#num_weeks = weeks_ddu_fcst
#
#output_loc="C:/Users/roysoumy/Documents/Demand Forecast Model/Results/"
DEMAND_LOC='//ant/dept/SortCenter/Network_Planning/Team/Soumya/'
def forecast_time_series(in_loc = '/local/home/roysoumy/data_files/ddu_inj_data_20200713.csv', num_weeks = 12, output_loc="C:/Users/roysoumy/Documents/Demand Forecast Model/Results/"):
        
    ##Reading zip mapping file "map_file_loc"
    print("Reading Mapping File")
    zip_map = pd.read_csv(in_loc, dtype = {"zip_code": str, "territory_code":str, "sort_center":str})
    print(str(zip_map.shape[0])+" records read.")
    
    zip_map['territory_code'] = zip_map['territory_code'].str.replace(' ','')
    zip_map['sort_center'] = zip_map['sort_center'].str.replace(' ','')
    zip_map['zip_code'] = zip_map['zip_code'].str.replace(' ','')
    ##Reading data from 2017 onwards	
    
    print("Reading historical data")
    data_2017 = pd.read_table(DEMAND_LOC +'2017_demand_data_by_zip.txt', 
                         skiprows = 1, delim_whitespace=True, names = ("ship_day", "zip_code", "pkgs"), dtype = {"ship_day": str, "zip_code":str, "pkgs":str})
    print("2017 data - "+str(data_2017.shape[0])+" records read.")
    
    data_2018 = pd.read_table(DEMAND_LOC +'2018_demand_data_by_zip.txt', 
                         skiprows = 1, delim_whitespace=True, names = ("ship_day", "zip_code", "pkgs"), dtype = {"ship_day": str, "zip_code":str, "pkgs":str})
    print("2018 data - "+str(data_2018.shape[0])+" records read.")
    
    data_2019 = pd.read_table(DEMAND_LOC +'2019_demand_data_by_zip.txt', 
                         skiprows = 1, delim_whitespace=True, names = ("ship_day", "zip_code", "pkgs"), dtype = {"ship_day": str, "zip_code":str, "pkgs":str})
    print("2019 data - "+str(data_2019.shape[0])+" records read.")
    
    data_2020 = pd.read_table(DEMAND_LOC +'2020_demand_data_by_zip.txt', 
                         skiprows = 1, delim_whitespace=True, names = ("ship_day", "zip_code", "pkgs"), dtype = {"ship_day": str, "zip_code":str, "pkgs":str})
    print("2020 data - "+str(data_2020.shape[0])+" records read.")
    
    data_2021 = pd.read_table(DEMAND_LOC+'2021_demand_data_by_zip.txt', 
                         skiprows = 1, delim_whitespace=True, names = ("ship_day", "zip_code", "pkgs"), dtype = {"ship_day": str, "zip_code":str, "pkgs":str})
    print("2021 data - "+str(data_2021.shape[0])+" records read.")

    ##Using latest upp data since packages to units ratio may vary sometimes and affect package level demand
    upp_data = pd.read_table(DEMAND_LOC+'upp_data_20200512.txt', 
                         skiprows = 1, delim_whitespace=True, names = ("ds", "upp"), dtype = {"ds": str, "upp":float})
    
    upp_data[['ds']] = upp_data[['ds']].apply(pd.to_datetime)
    
    ##Topline data as projected by NA Trans or other teams based on their execution level estimates
    topline_data = pd.read_table(DEMAND_LOC+'topline_data_weekly.txt', 
                         skiprows = 1, delim_whitespace=True, names = ("ds", "topline"), dtype = {"ds": str, "topline":float})
    
    topline_data[['ds']] = topline_data[['ds']].apply(pd.to_datetime)
    
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['text.color'] = 'G'
    ##Getting list of zips that are present in "map_file_loc"
    zip_info = r"[0-9]{5}"
    
    
    zip_list = zip_map['zip_code'].unique()
    
    ##Filtering zip level demand data for only zips that are present in "map_file_loc"
    data_2017 = data_2017[data_2017['zip_code'].isin(zip_list)]
    data_2018 = data_2018[data_2018['zip_code'].isin(zip_list)]
    data_2019 = data_2019[data_2019['zip_code'].isin(zip_list)]
    data_2020 = data_2020[data_2020['zip_code'].isin(zip_list)]
    data_2021 = data_2021[data_2021['zip_code'].isin(zip_list)]
    
    ##Cleaning data for 2017
    #data_2017 = data_2017[~data_2017['zip_code'].str.contains("[a-zA-Z]").fillna(False)]
    data_2017 = data_2017[~data_2017['pkgs'].str.contains("[a-zA-Z]").fillna(False)]
    
    data_2017[['ship_day']] = data_2017[['ship_day']].apply(pd.to_datetime)
    data_2017[['pkgs']] = data_2017[['pkgs']].apply(pd.to_numeric)
    
    ##Cleaning data for 2018
    #data_2018 = data_2018[~data_2018['zip_code'].str.contains("[a-zA-Z]").fillna(False)]
    data_2018 = data_2018[~data_2018['pkgs'].str.contains("[a-zA-Z]").fillna(False)]
    
    data_2018[['ship_day']] = data_2018[['ship_day']].apply(pd.to_datetime)
    data_2018[['pkgs']] = data_2018[['pkgs']].apply(pd.to_numeric)
    
    ##Cleaning data for 2019
    #data_2019 = data_2019[~data_2019['zip_code'].str.contains("[a-zA-Z]").fillna(False)]
    data_2019 = data_2019[~data_2019['pkgs'].str.contains("[a-zA-Z]").fillna(False)]
    
    data_2019[['ship_day']] = data_2019[['ship_day']].apply(pd.to_datetime)
    data_2019[['pkgs']] = data_2019[['pkgs']].apply(pd.to_numeric)

    ##Cleaning data for 2019
    #data_2019 = data_2019[~data_2019['zip_code'].str.contains("[a-zA-Z]").fillna(False)]
    data_2020 = data_2020[~data_2020['pkgs'].str.contains("[a-zA-Z]").fillna(False)]
    
    data_2020[['ship_day']] = data_2020[['ship_day']].apply(pd.to_datetime)
    data_2020[['pkgs']] = data_2020[['pkgs']].apply(pd.to_numeric)

    data_2021 = data_2021[~data_2021['pkgs'].str.contains("[a-zA-Z]").fillna(False)]
    
    data_2021[['ship_day']] = data_2021[['ship_day']].apply(pd.to_datetime)
    data_2021[['pkgs']] = data_2021[['pkgs']].apply(pd.to_numeric)
             
    del zip_info

    del zip_list	
    
    ##Appending all data
    data_final = data_2017.append(data_2018).append(data_2019).append(data_2020).append(data_2021)
    
    data_final_v2 = pd.merge(data_final, zip_map, how = 'inner', on = 'zip_code')
    data_final_v2[['pkgs']] = data_final_v2[['pkgs']].apply(pd.to_numeric)
    data_final_v2[['ship_day']] = data_final_v2[['ship_day']].apply(pd.to_datetime)
    
    ##Creating data "key" to roll up data (key - Sort Center_Territory)
    data_final_v2['time_series_key'] = data_final_v2['sort_center']+'_'+data_final_v2['territory_code']

    ##Data rollup to "key" level
    data_final_v4 = data_final_v2[['ship_day', 'time_series_key', 'pkgs']].groupby(['ship_day', 'time_series_key'], as_index=False).agg({"pkgs": "sum"})
    
    #Converting to week start date - Sunday date of the week
    data_final_v4['ship_day'] = pd.to_datetime(data_final_v4['ship_day'])  
    # 'daysoffset' will container the weekday, as integers
    wk_to_date = lambda a: a - dt.timedelta(days=a.isoweekday() % 7)
    
    data_final_v4['week_start'] = data_final_v4['ship_day'].apply(wk_to_date)

    ## added by pravengs@
    data_final_v4['week_start']=data_final_v4['week_start'].apply(lambda x: x.date())
    
    ##Rolling up data to weekly level to avoid higher variability by day issue
    ##Using daily data creates it difficult for model to forecast at higher accuracy, weekly better
    data_final_v4 = data_final_v4[['week_start', 'time_series_key', 'pkgs']].groupby(['week_start', 'time_series_key'], as_index=False).agg({"pkgs": "sum"})
    
    ##Getting 7 days' old filter to filter data to last week
    today = dt.date.today()- dt.timedelta(days = 7)

    
    data_final_v5 = data_final_v4[data_final_v4.week_start < today]
    
    unique_terr = data_final_v5['time_series_key'].unique()
    
    data_final_v5[['week_start']] = data_final_v5[['week_start']].apply(pd.to_datetime)
    
    ###Dates of prime week, thansgiving and peak week
    special_dates = ['2017-07-09', '2017-11-19', '2017-11-26', '2017-12-17', '2018-07-15', '2018-11-18', '2018-11-25', '2018-12-16', '2019-07-14', '2019-11-24', '2019-12-01', '2019-12-15', '2020-09-13', '2020-11-22', '2020-11-29', '2020-12-13', '2021-11-21', '2021-12-19']
    ##data_final_v6 = data_final_v6[['week_start', 'pkgs', 'time_series_key', 'holiday']]
    
    data_final_v5.columns =  ['ds', 'time_series_key', 'y']
  
    data_final_v5 = pd.merge(data_final_v5, upp_data, how = 'left', on = 'ds')
    
    data_final_v5 = pd.merge(data_final_v5, topline_data, how = 'left', on = 'ds')
    
    forecast_data = pd.DataFrame(columns = ['Week_Start_Date', 'Packages', 'Forecast_Type', 'time_series_key'])
    
       
    print("Forecasting begins for - ")
    
    ##Forecasting for each "key"
    for x in unique_terr:
        #x = 'STL5_STL5'
        print(x)
        #holidays = data_final_v5[data_final_v5.time_series_key == x]
        holidays = data_final_v5[data_final_v5.time_series_key == x]
        

        ###SARIMAX Model begins
        df = holidays[['ds', 'y']]
        df.columns = ['Date', 'Val']
        y = df.set_index(['Date'])
        #y.head(5)

        #plt.plot(y)

        data2 = holidays[['ds', 'topline']]
        data2.columns = ['Date', 'topline']
        data2 = data2.set_index(['Date'])

        #plt.plot(data2)
        
        ##Tuning SARMAX and choosing best model based on AIC and BIC (these penalize based on fit and number of predictors)
        #p = d = q = range(0, 2)
        #pdq = list(itertools.product(p, d, q))
        #seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
        #print('Examples of parameter for SARIMA...')
        #print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
        #print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
        #print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
        #print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
        #
        #for param in pdq:
        #    for param_seasonal in seasonal_pdq:
        #        try:
        #            mod = sm.tsa.SARIMAX(y, exog = data2, order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
        #            results = mod.fit()
        #            print('ARIMA{}x{} - AIC:{}'.format(param,param_seasonal,results.aic))
        #        except: 
        #            continue



        mod1 = sm.tsa.SARIMAX(y, exog = data2, order = (1, 1, 1), seasonal_order = (1, 1, 0, 52), enforce_stationarity=False,enforce_invertibility=False)
        res = mod1.fit()
        #print(res.summary())
        #res.plot_diagnostics()



        sar_check = res.predict(exog = data2)

        sar_check = pd.DataFrame(sar_check)
        sar_check.reset_index(inplace = True)
        sar_check.columns = ["ds" , "yhat_sar"]
        ##SARIMAX Model ends

        ###Prophet begins
        holidays = holidays[['ds', 'y', 'upp', 'topline']]
        
        holidays['special_events'] = holidays['ds'].isin(special_dates)
        holidays['upp'] = holidays['upp']
        holidays['topline'] = holidays['topline']
        
    
        #holidays = holidays.reset_index()
        
        ###Using Prophet model - adding other factors - upp, topline units projection, special amazon events
        m = Prophet(weekly_seasonality = False)
        m.add_regressor('special_events')
        m.add_regressor('upp')
        m.add_regressor('topline')
        
        m.add_seasonality(name = 'weekly', period = 52, fourier_order = 10)
        
        m.fit(holidays)
        

        
    
        future = m.make_future_dataframe(periods = num_weeks, freq = 'W')
        future['special_events'] = future['ds'].isin(special_dates)
        future = pd.merge(future, upp_data, how = 'left', on = 'ds')
        future = pd.merge(future, topline_data, how = 'left', on = 'ds')
        fcst = m.predict(future)
        ###Prophet ends
        
        ##Getting SARIMAX model forecast for user defined weeks
        sar_data2 = future[['topline']].tail(fcst.shape[0]-holidays.shape[0])
        
        pred = res.get_forecast(steps = num_weeks, exog = sar_data2)
        sar_pred = pred.predicted_mean

        sar_pred = pd.DataFrame(sar_pred)

        sar_pred.reset_index(inplace = True)
        sar_pred.columns = ["ds" , "y"]

        ###Holt Winters Exponential Smoothing begins
        mod_holt = ExponentialSmoothing(y, seasonal_periods = 52, trend = 'add', seasonal = 'add')
        mod_holt_fit = mod_holt.fit()
        #print(mod_holt_fit.summary())

        holt_pred = mod_holt_fit.forecast(steps = num_weeks)
        holt_pred = pd.DataFrame(holt_pred)
        holt_pred.reset_index(inplace = True)
        holt_pred.columns = ["ds" , "y"]

        holt_check = mod_holt_fit.fittedvalues
        holt_check = pd.DataFrame(holt_check)
        holt_check.reset_index(inplace = True)
        holt_check.columns = ["ds" , "yhat_holt"]


        
        ##Choosing best model
        ##Creating a correction factor because sometimes the model doesnt recognize small changes in network configuration that actually decreases or increases demand
        check_factor_data = pd.DataFrame(holidays.iloc[(int(holidays.shape[0])-3):holidays.shape[0]])
        fcst_check = fcst[['ds', 'yhat']]
        
        check_factor_data = pd.merge(check_factor_data, fcst_check, how = "inner", on = "ds")
        
        check_factor_data = pd.merge(check_factor_data, sar_check, how = "inner", on = "ds")
        
        check_factor_data = pd.merge(check_factor_data, holt_check, how = "inner", on = "ds")
        
        check_factor_data = check_factor_data.agg({'y':['sum'], 'yhat':['sum'], 'yhat_sar':['sum'], 'yhat_holt':['sum']})
        
        check_factor_data['check_factor_pp'] = check_factor_data['y']/check_factor_data['yhat']
        check_factor_data['check_factor_sar'] = check_factor_data['y']/check_factor_data['yhat_sar']
        check_factor_data['check_factor_holt'] = check_factor_data['y']/check_factor_data['yhat_holt']
        
        check_factor_pp = float(check_factor_data['check_factor_pp'].iloc[0])
        check_factor_sar = float(check_factor_data['check_factor_sar'].iloc[0])
        check_factor_holt = float(check_factor_data['check_factor_holt'].iloc[0])

        abs_list = (abs(check_factor_pp-1), abs(check_factor_sar-1), abs(check_factor_holt-1))
        
        check_factor_list = (check_factor_pp, check_factor_sar, check_factor_holt)
        
        check_factor = abs_list.index(min(abs_list))
        check_factor = float(check_factor_list[check_factor])
        
        ##Choosing the best model and using appropriate forecast
        if (check_factor == check_factor_pp):
            print(str("Prophet Model is chosen for "+x))
            fcst_final = fcst[['ds', 'yhat']].tail(fcst.shape[0]-holidays.shape[0])
            fcst_final.columns = ['ds', 'y']
        elif (check_factor == check_factor_sar):
            print(str("SARIMAX Model is chosen for "+x))
            fcst_final = sar_pred
            fcst_final.columns = ['ds', 'y']
        else:
            print(str("Holt-Winters Model is chosen for "+x))
            fcst_final = holt_pred
            fcst_final.columns = ['ds', 'y']
      

          
           
    
    
        ##Now patching up the forecast values with the actuals from history
        #fcst_final = fcst[['ds', 'yhat']].tail(fcst.shape[0]-holidays.shape[0])
        #fcst_final = fcst[['ds', 'yhat']].tail(fcst.shape[0]-holidays.shape[0])
        #fcst_final.columns = ['ds', 'y']
        
        holidays['forecast_type'] = 'Actuals'
        
        fcst_final['forecast_type'] = 'Forecast'
        
        holidays = holidays[['ds', 'y', 'forecast_type']]
        
        
 
        
        fcst_final = pd.concat([holidays, fcst_final])
        
        fcst_final.columns = ['Week_Start_Date', 'Packages', 'Forecast_Type']
        
        ###Scaling peak data since model does not account for UPP decrease or if advised by upper management to check for case if scenarios
#        def pkg_scale(x):
#            if x['Forecast_Type'] == 'Forecast' and x['Week_Start_Date'].month == 12 and x['Week_Start_Date'].strftime('%Y-%m-%d') in special_dates:
#                val = x['Packages']*1.05
#            elif x['Forecast_Type'] == 'Forecast' and x['Week_Start_Date'].strftime('%Y-%m-%d') in special_dates:
#                val = x['Packages']*1.00
#            else:
#                val = x['Packages']*1.00
#            return val
        
        
        
        
        print(fcst_final.shape)
        #fcst_final['Packages_Forecasted'] = fcst_final.apply(pkg_scale, axis = 1)
        fcst_final['Packages_Forecasted'] = fcst_final['Packages']

        fcst_final = fcst_final[['Week_Start_Date', 'Packages_Forecasted', 'Forecast_Type']]
        fcst_final.columns = ['Week_Start_Date', 'Packages', 'Forecast_Type']
        
        
        fcst_final['time_series_key'] = x
        
        
        forecast_data = pd.concat([forecast_data, fcst_final])
        
    ###Getting sort center and territory columns from time series key        
    forecast_data['sort_center'] = forecast_data.time_series_key.str[:4]
    
    forecast_data['territory'] = forecast_data.time_series_key.str[5:]
    
    #print("Writing Forecasted data to output location - /local/home/roysoumy/data_files/")
    forecast_data.to_csv(str(output_loc+"Demand_Forecast_"+str(dt.date.today())+".txt"), index = False)
    print('Forecast Done.')

    #print('Holt Winters output filename - Forecast_output_Holt_Winters.txt')
    
    ###Plot for Demand Forecast
#    forecast_data_2 = forecast_data.groupby(["Week_Start_Date"], as_index = False).agg({'Packages':'sum'})
#    
#    plt.plot(forecast_data_2["Week_Start_Date"], forecast_data_2["Packages"], "black", label = "Demand Forecast", linewidth = 6)
#    plt.xlabel('Week Start Date')
#    plt.ylabel('Demand')
#    plt.title('Demand Forecast')
#    plt.legend()
#    plt.grid(True, color = 'green')
#    
#    plt.savefig(str(output_loc+'demand_forecast.png'), bbox_inches = 'tight')
#    plt.close()
    return forecast_data


#forecast_time_series(in_loc = map_file_loc, num_weeks = weeks_ddu_fcst)