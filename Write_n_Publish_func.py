# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:32:11 2020

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


from matplotlib import pyplot as plt
from matplotlib import style
import holidays 
import os



##Function to write a dataframe to required database and write to file
#def write_and_publish(ip_df, date_of_launch, write_file_loc, publish_y_n, publish_table, host, port, user, password, db):
def write_and_publish(ip_df, date_of_launch, write_file_loc, publish_y_n, publish_table, host, port, user, password, db):    
    ##Forecast starts from first day of current week but days before today are not forecast and hence filtered
    today_date = dt.date.today()
    final_data3 = ip_df.loc[ip_df['inbound_date'] > today_date]
    
	##For secnarios where we want to see numbers only from launch date, filter
    launch_date = pd.to_datetime(date_of_launch)
    if launch_date > today_date :
        final_data3 = final_data3.loc[final_data3['inbound_date'] >= launch_date]
    
    ##Writing to desired location
    final_data3.to_csv(str(write_file_loc+ "DDU_Forecast_" +str(dt.date.today())+"_daily.txt"), sep = "\t", index = False)
    print("DDU forecast written to file!")
    
    ##Writing the file to database, default is forecast database with proper format
    import pymysql

    rs_config_mysql={'host':host,'port':port,'user':user,'passwd':password,'database':db}

	
    rs_conn_str_mysql='mysql+pymysql://'+rs_config_mysql['user']+':'+rs_config_mysql['passwd']+'@'+rs_config_mysql['host']+':'+rs_config_mysql['port']+'/'+rs_config_mysql['database']


    rs_db_mysql = create_engine(rs_conn_str_mysql)

    final_data4 = final_data3

    final_data4['publish_date']  = str(dt.date.today())
    
    ##For scenarios where publishing to database is not required
    if publish_y_n == "Yes":
        final_data4.to_sql(name=publish_table, con=rs_db_mysql, if_exists = 'append', index=False)
        print("DDU Data published to database!")
    
    
    return(final_data3)