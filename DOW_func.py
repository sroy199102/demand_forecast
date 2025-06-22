# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:21:53 2020

@author: roysoumy
"""
import numpy as np
import pandas as pd
import re
import datetime as dt
from csv import reader
import statsmodels.api as sm
from sklearn import linear_model
import os

from fbprophet import Prophet

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sqlalchemy import create_engine


from matplotlib import pyplot as plt
from matplotlib import style
import holidays 

USER='pravengs'
PASSWD=os.getenv('SCREDSHIFT_PASSWD')


def dow(location = "default", f_days = 91, op_loc="C:/Users/roysoumy/Documents/Demand Forecast Model/Results/"):
    
    def mapping():
        rs_config = {
        'host':"sc-redshift-instance.cjnte6rjhunx.us-east-1.redshift.amazonaws.com"
        ,'port':"8192"
        ,'user':USER#'roysoumy'
        ,'passwd':PASSWD#'IRe$erve8'
        ,'database':"scredshift"
        }

        rs_conn_str = 'postgresql://'+rs_config['user']+':'+rs_config['passwd']+'@'+rs_config['host']+':'+rs_config['port']+'/'+rs_config['database']
        rs_db = create_engine(rs_conn_str)
        mapping_query = '''SELECT zipcode as zip_code, sort_center as territory_code, sort_center FROM network_opt.ddu_mapping_midterm
        WHERE publish_date = (SELECT MAX(publish_date) FROM network_opt.ddu_mapping_midterm) and phase = 'Postpeak2020' and start_date = (SELECT MAX(start_date) FROM network_opt.ddu_mapping_midterm) '''
     
        map = pd.read_sql(mapping_query, con=rs_db)
    
        return map
    
    if location == "default":
        mapping = mapping()
    else:
        mapping = pd.read_csv(location, dtype = {"zip_code": str, "territory_code":str, "sort_center":str})
        
    mapping = mapping[['zip_code', 'sort_center']]
    
    
    mapping['sort_center'] = mapping['sort_center'].str.replace(' ','')
    mapping['zip_code'] = mapping['zip_code'].str.replace(' ','')
    
    ##Quality check to see what sort centers are present in the mapping file
    scs = mapping['sort_center'].unique()
    print("The sort centers in the mapping file are -")
    print(scs)

    ## get ddu actual data by zip 
    rs_config = {
        'host':"sc-redshift-instance.cjnte6rjhunx.us-east-1.redshift.amazonaws.com"
        ,'port':"8192"
        ,'user':USER#'roysoumy'
        ,'passwd':PASSWD#'IRe$erve8'
        ,'database':"scredshift"
        }

    rs_conn_str = 'postgresql://'+rs_config['user']+':'+rs_config['passwd']+'@'+rs_config['host']+':'+rs_config['port']+'/'+rs_config['database']
    rs_db = create_engine(rs_conn_str)
    
    ddu_query = '''SELECT lpad(substring(shipping_address_postal_code, 1, 5), 5, '0') AS zip_code
      ,'DDU' AS ship_method
      ,trunc(coalesce(IB_ACTUAL_ARRIVAL_TIME_LOCAL, IB_SCHEDULED_ARRIVAL_TIME_LOCAL, FIRST_SCAN_IN_SC_LOCAL)) AS inbound_date
      ,count(DISTINCT fulfillment_shipment_id || package_id) AS pkgs
    FROM booker.d_nasc_pkg_data
    WHERE region_id = 1
      AND trunc(coalesce(IB_ACTUAL_ARRIVAL_TIME_LOCAL, IB_SCHEDULED_ARRIVAL_TIME_LOCAL, FIRST_SCAN_IN_SC_LOCAL)) BETWEEN sysdate - 14 - 14
        AND sysdate - 1
      AND (
        ship_method LIKE '%%USPS_SC%%'
        OR ship_method LIKE '%%USPS_ATS%%'
        )
      AND substring(ob_lane, 7, 4) NOT IN (
        select distinct sort_center 
        from booker.dim_sort_center_list
        )
    GROUP BY lpad(substring(shipping_address_postal_code, 1, 5), 5, '0')
      ,ship_method
      ,trunc(coalesce(IB_ACTUAL_ARRIVAL_TIME_LOCAL, IB_SCHEDULED_ARRIVAL_TIME_LOCAL, FIRST_SCAN_IN_SC_LOCAL))
      '''
    
    ddu_a = pd.read_sql(ddu_query, con=rs_db)
    
    ddu_a['inbound_date'] = pd.to_datetime(ddu_a['inbound_date'])
    
    # get ddu actual data for 2019
    
    #ddu_b = pd.read_csv("//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/ddu_zip_actual_2019.csv", dtype={'zip_code':'object'})
    #ddu_b = pd.read_csv("//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/ddu_zip_actual_2020.csv", dtype={'zip_code':'object'})
    ddu_b = pd.read_csv("//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/ddu_zip_actual_2021.csv"
            , sep='\t'
            ,dtype={'zip_code':'object'})
    
    ddu_b['inbound_date'] = pd.to_datetime(ddu_b['inbound_date'])
    max_date_ddu_b=ddu_b['inbound_date'].max()
    ddu_a1 = pd.concat([ddu_b, ddu_a.query('inbound_date>@max_date_ddu_b')])
    
    # remove NA values
    
    ddu_a1 = ddu_a1.dropna()
    #print ("ddu_a",ddu_a.inbound_date.min(), ddu_a.inbound_date.max())
    #print ("ddu_b",ddu_b.inbound_date.min(), ddu_b.inbound_date.max())

    #print (ddu_a1.inbound_date.min(), ddu_a1.inbound_date.max())
    
    # find out the week 
    
    #date_query = '''select caldate as inbound_date, week from booker.dim_calendar
    #where year between extract(year from (sysdate - 730)) and extract(year from (sysdate + 365))'''
    
    #date_1 = pd.read_sql(date_query, con=rs_db)
    
    #date_1['inbound_date'] = pd.to_datetime(date_1['inbound_date'])
    
    #print (date_1.inbound_date.min(), date_1.inbound_date.max())
    # get week from dim_calendar 
    
    #ddu_a2 = pd.merge(ddu_a1
    #                  , date_1
    #                  , on = ['inbound_date']
    #                  , how = 'left')
    
    ddu_a2=ddu_a1.copy()
    ddu_a2['week']=ddu_a2['inbound_date'].apply(lambda x:(x + dt.timedelta(1)).isocalendar()[1])
    #ddu_a2['year'] = pd.to_datetime(ddu_a2['inbound_date']).apply(lambda x:x.strftime('%Y'))
    ddu_a2['year'] = ddu_a2['inbound_date'].apply(lambda x:(x + dt.timedelta(1)).isocalendar()[0])
    
    #print ("ddu_a2_year",ddu_a2.year.value_counts())
    # get number of dates in every week
    ddu_a2['year']=ddu_a2['year'].apply(lambda x: int(x))
    ddu_a2_tmp = ddu_a2.groupby(['zip_code', 'year', 'week']).nunique()['inbound_date'].reset_index().query('inbound_date == 7').rename(columns={'inbound_date':'n_distinct'})
    
    #print ("ddu_a2 sample 2021",ddu_a2.query('year =="2021"').head(10))
    #print ("ddu_a2 sample 2020",ddu_a2.query('year =="2020"').head(10))
    #print ("ddu_a2 dtypes", ddu_a2.dtypes)
    #print ("ddu_a2 min, max", ddu_a2.inbound_date.min(), ddu_a2.inbound_date.max())
    #print ("ddu a2 group",ddu_a2.groupby(['year','week']).agg({'inbound_date':
    #    [np.min,np.max,pd.Series.nunique]
    #    ,'zip_code':['count']
    #    }))
    
    # keep dates forming complete week 
    #print (ddu_a2_tmp.head())
    #print ("ddu_a2 nan count",ddu_a2.isna().sum())
    #print ("ddu_a2_year",ddu_a2.year.value_counts())
    #print ("ddu_a2_tmp_year",ddu_a2_tmp.year.value_counts())

    ddu_a2_tmp2 = pd.merge(ddu_a2
                      , ddu_a2_tmp
                      , on = ['zip_code', 'year', 'week']
                      , how = 'inner')
    
    # merge mapping with actuals 
    #print ("ddu_a2_tmp2",ddu_a2_tmp2.columns)
    #print ("ddu_a2_tmp2_year",ddu_a2_tmp2.year.value_counts())
    #print ("mapping",mapping.columns)
    
    ddu_a3 = pd.merge(ddu_a2_tmp2
                      , mapping
                      , on = ['zip_code']
                      , how = 'inner')
      
    # calculate week sum 
    
    ddu_a4 = ddu_a3.groupby(['sort_center', 'week', 'year']).sum()['pkgs'].reset_index().rename(columns={'pkgs':'week_sum'})
    
    # merge the total 

    #print ("ddu_a3",ddu_a3.columns, "ddu_a4",ddu_a4.columns)
    #print ("ddU_a3",ddu_a3.year.value_counts(), "ddu_a4", ddu_a4.year.value_counts())
    
    ddu_a5 = pd.merge(ddu_a3
                      , ddu_a4
                      , on = ['sort_center', 'week', 'year']
                      , how = 'inner')
    
    # calculate the dow 
    
    dow_1 = ddu_a5.groupby(['sort_center', 'inbound_date', 'week', 'year']).agg({'pkgs':[np.sum],
                          'week_sum':[np.mean]}).reset_index()
    
    dow_1.columns = ['_'.join(str(s).strip() for s in col if s) for col in dow_1.columns]
    
    dow_1['dow'] = dow_1['pkgs_sum'] * 1.0 / dow_1['week_sum_mean']
    
    dow_1 = dow_1.drop(['pkgs_sum', 'week_sum_mean'], axis = 1)
    
    # get weekday 
    
    dow_1['day'] =  dow_1['inbound_date'].apply(lambda x:x.strftime('%A'))
    
    # apply condition on dow to get latest dow using 42 days 
    
    D1 = dt.date.today() - dt.timedelta(42)
    dow_1a =  dow_1.query('inbound_date >= @D1')
    
    #print (dow_1.head(),dow_1.inbound_date.min(),dow_1.inbound_date.max() ,dow_1a.head())
    # get mean dow 
    
    dow_2 = dow_1a.groupby(['sort_center', 'day']).mean()['dow'].reset_index()
    
    # create next 3 months forecast dates 
    #print (dow_2.head())
    
    day_list = []
    start_date = dt.date.today()
    
    for i in range(f_days):
    	day = start_date + dt.timedelta(days = i)
    	day_list.append(day)
        
    dow_3 = pd.DataFrame({'inbound_date':day_list})
    
    dow_3['day'] =  dow_3['inbound_date'].apply(lambda x:x.strftime('%A'))
    
    # merge the forecast dates with dow 
    
    dow_4 = pd.merge(dow_2
                      , dow_3
                      , on = ['day']
                      , how = 'left')
    
    # convert to date type 
    
    dow_4['inbound_date'] = pd.to_datetime(dow_4['inbound_date'])
    # get all public holidays
    # get current year 
    
    currentYear = dt.date.today().year
    
    name_list = []
    date_list = []
    
    for date, name in sorted(holidays.US(years=currentYear-1,observed=False).items()):
        name_list.append(name)
        date_list.append(date)
        
    
    holidays_tmp = pd.DataFrame({'holiday':name_list
        ,'holiday_date':date_list})
    
    # get last year dow for holiday 
        
    
    holiday_ValentinesDay = pd.DataFrame({'holiday':['ValentinesDay']
                                        ,'holiday_date':[dt.datetime(currentYear-1,2,14).date()]})
        
    holiday_Halloween = pd.DataFrame({'holiday':['Halloween']
                                        ,'holiday_date':[dt.datetime(currentYear-1,10,31).date()]})
    
    holiday_StPatricksDay = pd.DataFrame({'holiday':['StPatricksDay']
                                        ,'holiday_date':[dt.datetime(currentYear-1,3,17).date()]})   
    
    holiday_CincodeMayo = pd.DataFrame({'holiday':['CincodeMayo']
                                        ,'holiday_date':[dt.datetime(currentYear-1,5,5).date()]})    
        
    def additional_holiday(fname,currentYear):
        tmp_df = pd.read_csv(fname,sep="\t")
    
        tmp_df['date_f'] = tmp_df['Date'] + ', '+ tmp_df['Year'].apply(lambda x:str(x))
        tmp_df['date_f1'] = pd.to_datetime(tmp_df['date_f']).apply(lambda x:x.date())  
    
        out_df = tmp_df[tmp_df['Year']==currentYear-1][['Name','date_f1']].rename(columns={'Name':'holiday'
                             ,'date_f1':'holiday_date'})
        return out_df
    
    holiday_easter = additional_holiday('//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/easter_holidays_2000_2049.txt',currentYear)
    holiday_MothersDay = additional_holiday('//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/mothers_day_holidays_2000_2049.txt',currentYear)
    holiday_FathersDay = additional_holiday('//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/fathers_day_holidays_2000_2049.txt',currentYear)
    holiday_PrimeDay = additional_holiday('//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/prime_day_holidays_2018_2029.txt',currentYear)
    
    
    # merge all holidays together
    
    holiday = pd.concat([holidays_tmp, holiday_ValentinesDay, holiday_Halloween, holiday_StPatricksDay, holiday_CincodeMayo,
                         holiday_easter, holiday_MothersDay, holiday_FathersDay, holiday_PrimeDay]) 
    
    # get last year holiday and nearby dates 
        
    holiday['h_prev'] = holiday['holiday_date'].apply(lambda x:x+dt.timedelta(-1))
    holiday['h_next'] = holiday['holiday_date'].apply(lambda x:x+dt.timedelta(1))
    
    # convert from wide to long format 
    
    id_vars = ['holiday']
    value_vars = ['holiday_date','h_prev','h_next']
    holiday_temp1 = pd.melt(holiday
                          ,id_vars=id_vars
                          ,value_vars=value_vars
                          ,value_name='inbound_date'
                          ).rename(columns={'variable':'holiday_day'})
    
    holiday_temp1['inbound_date'] = pd.to_datetime(holiday_temp1['inbound_date'])
    
    # get the dow for those weeks 
    
    dow_3['inbound_date'].apply(lambda x:x.strftime('%A'))
    dow_1['year'] = dow_1['year'].apply(lambda x:int(x))
    dow_ly = dow_1.query('year==(@currentYear - 1)')
    
    holiday_2 = pd.merge(holiday_temp1
                      , dow_ly
                      , on = ['inbound_date']
                      , how = 'inner')
    
    holiday_3 = holiday_2[['holiday', 'holiday_day', 'sort_center', 'dow']]
    
    # get current year holiday 
    
    name_list = []
    date_list = []
    
    for date, name in sorted(holidays.US(years=currentYear,observed=False).items()):
        name_list.append(name)
        date_list.append(date)
        
    
    holidays_tmp = pd.DataFrame({'holiday':name_list
        ,'holiday_date':date_list})
    
    # get this year dow for holiday 
        
    
    holiday_ValentinesDay = pd.DataFrame({'holiday':['ValentinesDay']
                                        ,'holiday_date':[dt.datetime(currentYear,2,14).date()]})
        
    holiday_Halloween = pd.DataFrame({'holiday':['Halloween']
                                        ,'holiday_date':[dt.datetime(currentYear,10,31).date()]})
    
    holiday_StPatricksDay = pd.DataFrame({'holiday':['StPatricksDay']
                                        ,'holiday_date':[dt.datetime(currentYear,3,17).date()]})   
    
    holiday_CincodeMayo = pd.DataFrame({'holiday':['CincodeMayo']
                                        ,'holiday_date':[dt.datetime(currentYear,5,5).date()]})    
        
    def additional_holiday(fname,currentYear):
        tmp_df = pd.read_csv(fname,sep="\t")
    
        tmp_df['date_f'] = tmp_df['Date'] + ', '+ tmp_df['Year'].apply(lambda x:str(x))
        tmp_df['date_f1'] = pd.to_datetime(tmp_df['date_f']).apply(lambda x:x.date())  
    
        out_df = tmp_df[tmp_df['Year']==currentYear][['Name','date_f1']].rename(columns={'Name':'holiday'
                             ,'date_f1':'holiday_date'})
        return out_df
    
    holiday_easter = additional_holiday('//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/easter_holidays_2000_2049.txt',currentYear)
    holiday_MothersDay = additional_holiday('//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/mothers_day_holidays_2000_2049.txt',currentYear)
    holiday_FathersDay = additional_holiday('//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/fathers_day_holidays_2000_2049.txt',currentYear)
    holiday_PrimeDay = additional_holiday('//ant/dept/SortCenter/Network_Planning/Team/Nikhil/datadump/prime_day_holidays_2018_2029.txt',currentYear)
    
    
    # merge all holidays together
    
    holiday_ty = pd.concat([holidays_tmp, holiday_ValentinesDay, holiday_Halloween, holiday_StPatricksDay, holiday_CincodeMayo,
                         holiday_easter, holiday_MothersDay, holiday_FathersDay, holiday_PrimeDay]) 
        
    # get last year holiday and nearby dates 
        
    holiday_ty['h_prev'] = holiday_ty['holiday_date'].apply(lambda x:x+dt.timedelta(-1))
    holiday_ty['h_next'] = holiday_ty['holiday_date'].apply(lambda x:x+dt.timedelta(1))
    
    # convert from wide to long format 
    
    id_vars = ['holiday']
    value_vars = ['holiday_date','h_prev','h_next']
    holiday_temp_ty1 = pd.melt(holiday_ty
                          ,id_vars = id_vars
                          ,value_vars = value_vars
                          ,value_name = 'inbound_date'
                          ).rename(columns = {'variable':'holiday_day'})
    
    holiday_temp_ty1['inbound_date'] = pd.to_datetime(holiday_temp_ty1['inbound_date'])
    
    # get the dow from last year for special days 
    
    holiday_dow = pd.merge(holiday_temp_ty1
                      , holiday_3
                      , on = ['holiday', 'holiday_day']
                      , how = 'inner')
    
    holiday_dow_1 = holiday_dow[['inbound_date', 'sort_center', 'dow']]
    
    holiday_dow_1 = holiday_dow_1.rename(columns = {"dow":"holiday_dow"}) 
    
    # merge with existing dow 
    dow_4['inbound_date']=dow_4['inbound_date'].apply(lambda x:pd.to_datetime(x,errors = 'coerce'))
    holiday_dow_1['inbound_date']=holiday_dow_1['inbound_date'].apply(lambda x:pd.to_datetime(x,errors = 'coerce'))
    
    #print (dow_4.dtypes,holiday_dow_1.dtypes )
    #print ("sort_centers",dow_4.sort_center.unique())
    
    dow_5 = pd.merge(dow_4
                      , holiday_dow_1
                      , on = ['inbound_date', 'sort_center']
                      , how = 'left')
    
    dow_5 = dow_5.fillna(0)
    
    def new_dow(row):
        if row['holiday_dow'] == 0:
            return row['dow']
        else:
            return row['holiday_dow']
        
    
    dow_5['new_dow'] = dow_5.apply(new_dow, axis = 1)
      
    dow_5['new_dow'] = dow_5.apply(lambda row: row['dow'] if row['holiday_dow']== 0 else row['holiday_dow'],axis = 1)  
        
    # get the week number 
    dow_5['inbound_date']=dow_4['inbound_date'].apply(lambda x:pd.to_datetime(x,errors = 'coerce'))
    #date_1['inbound_date']=date_1['inbound_date'].apply(lambda x:pd.to_datetime(x,errors = 'coerce'))

    #dow_6 = pd.merge(dow_5
    #                  , date_1
    #                  , on = ['inbound_date']
    #                  , how = 'left')

    dow_6=dow_5.copy()
    dow_6['week']=dow_6['inbound_date'].apply(lambda x:(x + dt.timedelta(1)).isocalendar()[1])
    dow_6['year']=dow_6['inbound_date'].apply(lambda x:(x + dt.timedelta(1)).isocalendar()[0])
    #dow_6['year'] = pd.to_datetime(dow_6['inbound_date']).apply(lambda x:x.strftime('%Y'))
    

    dow_9 = dow_6[['inbound_date', 'sort_center', 'day', 'new_dow']]
    
    dow_9.columns = ['inbound_date', 'sort_center', 'day', 'final_dow']
    
    wk_to_date = lambda a: a - dt.timedelta(days=a.isoweekday() % 7)

    dow_9['Week_Start_Date']= dow_9['inbound_date'].apply(wk_to_date)
    # return the output 
    print("Day of the week breakdown done!")
    
    dow_9.to_csv(str(op_loc+'DOW_'+str(dt.date.today())+".txt"), index = False)
    
    return(dow_9)

#a = dow(map_file_loc, dow_days)
#a.to_csv(r'C:/Users/roysoumy/Documents/Demand Forecast Model/Results/Dow_'+str(dt.date.today())+".txt", index = False)
