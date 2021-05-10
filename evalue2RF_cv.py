#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 08:33:52 2019

@author: fankai
"""

#pylab
import pylab
#pandas
import pandas as pd
print('pandas: %s' % pd.__version__)
#matplotlib
import matplotlib.pyplot as plt
#datetime
import datetime as dt
#import numpy
import numpy as np
#sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#gaussian_kde
from scipy.stats import gaussian_kde
#anchored text
from matplotlib.offsetbox import AnchoredText 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
import time
import pickle

#open functions
#exec(open(r"/Users/fankai/Research/ML/ML_Func_New.py").read())
exec(open(r"D:\Research\ML\ML_Func_New.py").read())
'''
# read csv files
#import data from previous 5 years
#df = pd.read_csv(r"/Users/fankai/Research/ML/All_Kennewick_and_Hermiston_O3_met_and_WRF_data_thru_2018.csv")
df = pd.read_csv(r"D:\Research\ML\All_Kennewick_and_Hermiston_O3_met_and_WRF_data_thru_2018.csv")

# set date from string to datetime and set as index and add dayofweek column
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
df['Weekday']=df['date'].dt.dayofweek
del df['date']

#obs_2019 = pd.read_csv('http://lar.wsu.edu/R_apps/2019ap5/data/byAQSID/530050003.apan',index_col=0)
obs_2019 = pd.read_csv('D:/Research/ML/530050003_2019.apan',index_col=0)
obs_2019.index = pd.to_datetime(obs_2019.index)
obs_2019 = obs_2019[['OZONEap','OZONEan']].dropna()
obs_2019.columns = ['Kennewick.OZONE_AIRPACT4','Kennewick.O3']
delta = np.timedelta64(8,'h')
obs_2019.index = obs_2019.index - delta
r = pd.date_range(start=obs_2019.index.min(), end=obs_2019.index.max(), freq='1H')
obs_2019 = obs_2019.reindex(r)
df = df.append(obs_2019)

#obs_2020 = pd.read_csv('http://lar.wsu.edu/R_apps/2020ap5/data/byAQSID/530050003.apan',index_col=0)
obs_2020 = pd.read_csv('D:/Research/ML/530050003_2020.apan',index_col=0)
obs_2020.index = pd.to_datetime(obs_2020.index)
obs_2020 = obs_2020[['OZONEap','OZONEan']].dropna()
obs_2020.columns = ['Kennewick.OZONE_AIRPACT4','Kennewick.O3']
delta = np.timedelta64(8,'h')
obs_2020.index = obs_2020.index - delta
r = pd.date_range(start=obs_2020.index.min(), end=obs_2020.index.max(), freq='1H')
obs_2020 = obs_2020.reindex(r)
df = df.append(obs_2020)

df_right = pd.read_csv('D:/Research/ML/Kennewick_WRF_from_AIRPACT.csv',index_col=0)
df_right.index = pd.to_datetime(df_right.index)
#df_right = df_right.iloc[:,list(range(4,9))+[11]]
df_right.columns = [
       'WRF_4km_BCAA.Surface_pres_Pa','WRF_4km_BCAA.PBL_m','WRF_4km_BCAA.T_K',
       'WRF_4km_BCAA.windspeed_m_per_s','WRF_4km_BCAA.WindDir_deg', 
       'WRF_4km_BCAA.RH_pct']
df.update(df_right)
#df = df[df.index.isin(df_right.index)]

df = df[(df.index.year>2016) & (df.index.month>4) & (df.index.month<10)]

#replace airnow obs with aqs
aqs = pd.read_csv('D:/Research/ML/AQS_O3_Kennewick.csv')
aqs.index = pd.to_datetime(aqs.time)
df = pd.concat([df,aqs],axis=1).drop(columns=['time','Kennewick.O3'])
df = df.rename(columns={"obs": "Kennewick.O3"})

#convert wind direction and spd to U and V components
#df['U']=-df['Windspeed_knots.Pasco_airport']*np.sin(df['Wind_dir_deg.Pasco_airport']*np.pi/180)
#df['V']=-df['Windspeed_knots.Pasco_airport']*np.cos(df['Wind_dir_deg.Pasco_airport']*np.pi/180)
#df['U']=-df['WRF_4km_BCAA.windspeed_m_per_s']*np.sin(df['WRF_4km_BCAA.WindDir_deg']*np.pi/180)
#df['V']=-df['WRF_4km_BCAA.windspeed_m_per_s']*np.cos(df['WRF_4km_BCAA.WindDir_deg']*np.pi/180)
#df['U']=df['WRF_4km_BCAA.windspeed_m_per_s']
#df['V']=df['WRF_4km_BCAA.WindDir_deg']
#df.loc[df['V']>180,'V'] -= 360
#df['U']=-df['WRF_4km_BCAA.windspeed_m_per_s']*np.sin(df['WRF_4km_BCAA.WindDir_deg']*np.pi/180)
#df['WindSector']=(df['WRF_4km_BCAA.WindDir_deg'].values/ 22.5 + 0.5).astype(int)
#df['WindSector'].loc[df['WindSector']<0] = np.nan
'''

df = pd.read_csv('D:/Research/ML/Kennewic_2017_2020.csv')
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']

#make a column of 8hr ozone
#calculate 8hr avg and shift
df['O3avg8hr'] = df['Kennewick.O3'].rolling(8, min_periods=6).mean()
df['O3avg8hr.ap'] = df['Kennewick.OZONE_AIRPACT4'].rolling(8, min_periods=6).mean()
#df['O3avg8hr'] = df['O3avg8hr'].shift(-7)
df['AQI_class'] = pd.cut(round(df['O3avg8hr'].fillna(-1)),
    [0, 54, 70, 85, 105, 200, np.inf],
    labels=[1, 2, 3, 4, 5, 6])


df['month'] = df.index.month
df['hour'] = df.index.hour
df['Weekday']=df.index.dayofweek

#make a column of previous days max ozone
df['past_O3_24'] = df['Kennewick.O3'].shift(24)

df['Wildfire_smoke_caused_excess_ozone_YN'][df['Wildfire_smoke_caused_excess_ozone_YN']=='Y'] = 1
df['Wildfire_smoke_caused_excess_ozone_YN'][df['Wildfire_smoke_caused_excess_ozone_YN']=='N'] = 0

#reduce the data of a day to 8am-7pm *currently issue with the truth value obs/model < or > 55
#df=df[(df.index.hour<=19)]
#df=df[(df.index.hour>=8)]

#exclude the wildfire affected days (only 0.37%)
df=df[(df['Wildfire_smoke_caused_excess_ozone_YN']!=1)]

delta = np.timedelta64(7,'h')
df.index = df.index - delta
    
#pasco met values only for previous 5 years data
dfPm = df[['Kennewick.O3', 'Kennewick.OZONE_AIRPACT4', 'WRF_4km_BCAA.PBL_m',
       'WRF_4km_BCAA.Surface_pres_Pa',
       #'WRF_4km_BCAA.Water_vap_mixing_ratio_kg_per_kg', 
       'WRF_4km_BCAA.T_K',
       'WRF_4km_BCAA.windspeed_m_per_s', 'WRF_4km_BCAA.WindDir_deg',
       'WRF_4km_BCAA.RH_pct', #'Wildfire_smoke_caused_excess_ozone_YN',
       'month','hour','Weekday','past_O3_24'#,'past_O3_18','past_O3_12','past_O3_6'
       ]].copy()

#dfPm['WRF_4km_BCAA.Surface_pres_Pa'] = dfPm['WRF_4km_BCAA.Surface_pres_Pa']/100
#dfPm['WRF_4km_BCAA.T_K'] = 9/5*(dfPm['WRF_4km_BCAA.T_K'] - 273) + 32

#rename columns to be more generic
dfPm.columns = ['O3_obs', 'O3_mod', 'PBL_m', 'Sea_level_Pressure_Pa', #'WVMR', 
                'Temp_K', 'WindSpeed_m_per_s', 'WindDir_deg',
                'RH_pct', #'FireSmoke', 
                'Month','Hour','Weekday','past_O3_24'#,'past_O3_18','past_O3_12','past_O3_6'
                ]

dfPm_org = dfPm.copy()
dfPm_org['O3avg8hr'] = dfPm_org['O3_obs'].rolling(8, min_periods=6).mean()

#dfPm = dfPm[dfPm.index.year>2015]

#%%
dfPm = dfPm.dropna().copy()
#dfPm = dfPm[(dfPm.index<'2018-08-16 07:00:00') | (dfPm.index>'2018-08-17 06:00:00')] #remove AQI 4 day

X_org = dfPm.drop(['O3_obs', 'O3_mod'], 1).copy()
X_dat = pd.DataFrame(preprocess('MAS', X_org))
X_dat.columns = X_org.keys()
X_dat.index = X_org.index

X = np.array(X_dat.copy())
# separate "label" to y 
y = np.array(dfPm['O3_obs'])

#test
#X = np.array(sorted(set(subdf.index.date)))
#rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=12883823)
#for train_index,test_index in rkf.split(X):
#    tmp = subdf[pd.to_datetime(subdf.index.date).isin(X[test_index])]
#    print(tmp['Weekday'].value_counts().sort_index())
    
all_date = np.array(sorted(set(dfPm.index.date)))
rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=12883823)
a = 0
dict_of_2019=dict()
dict_of_max2019=dict()
RF1_feature_table = pd.DataFrame(columns = dfPm.keys()[2:]) 
RF2_feature_table = pd.DataFrame(columns = dfPm.keys()[2:])
for train_index, test_index in rkf.split(all_date):
    train_datetime_index = pd.to_datetime(dfPm.index.date).isin(all_date[train_index])
    test_datetime_index = pd.to_datetime(dfPm.index.date).isin(all_date[test_index])
    X_train, X_test = X[train_datetime_index], X[test_datetime_index]
    y_train, y_test = y[train_datetime_index], y[test_datetime_index]
    
    # feature extraction
    model_RF = RandomForestRegressor(n_estimators=100, max_depth=7,random_state=137)
    model_RF = model_RF.fit(X_train, y_train)
    
    new_train = pd.DataFrame(X_train)
    new_train['obs_o3'] = y_train
    new_train['pred_o3'] = model_RF.predict(X_train)
    new_train['diff_o3'] = abs(new_train['pred_o3']-new_train['obs_o3'])
    new_train=new_train[new_train['diff_o3']>5]
    X_train2 = np.array(new_train.drop(['obs_o3','pred_o3','diff_o3'],axis=1))
    y_train2 = y_train[new_train.index]
    model_RF2 = RandomForestRegressor(n_estimators=200, max_depth=7,random_state=137)
    model_RF2 = model_RF2.fit(X_train2, y_train2)
    
    RF1_feature_table = RF1_feature_table.append(dict(zip(RF1_feature_table.columns, model_RF.feature_importances_)), ignore_index=True)
    RF2_feature_table = RF2_feature_table.append(dict(zip(RF2_feature_table.columns, model_RF2.feature_importances_)), ignore_index=True)
    #RF1_feature_table.to_csv(
    #    '/Users/fankai/Research/ML/ML2_cv_RF1_feature_noAQI4.csv', index=False)
    #RF2_feature_table.to_csv(
    #    '/Users/fankai/Research/ML/ML2_cv_RF2_feature_noAQI4.csv', index=False)
    RF1_feature_table.to_csv('D:\Research\ML\cross_validation_aqs\ML2_cv_RF1_feature.csv',index=False)
    RF2_feature_table.to_csv('D:\Research\ML\cross_validation_aqs\ML2_cv_RF2_feature.csv',index=False)
    
    '''
    low_index = new_train[new_train['pred_o3']<40].index
    med_index = new_train[(new_train['pred_o3']<=60) & (new_train['pred_o3']>=40)].index
    high_index = new_train[new_train['pred_o3']>60].index
    '''
    pred1=model_RF.predict(X_train)
    pred2=model_RF2.predict(X_train)
    
    sep1 = np.percentile(pred1,33)
    sep2 = np.percentile(pred1,67)
    
    low_index = np.where(pred1<sep1)
    med_index = np.where((pred1<=sep2) & (pred1>=sep1))
    high_index = np.where(pred1>sep2)
    
    X_low=list (zip (pred1[low_index],pred2[low_index]))
    Y_low=y_train[low_index]
    model_LR = LinearRegression(fit_intercept=False)
    model_LR.fit(X_low, Y_low)
    low1=model_LR.coef_[0]
    low2=model_LR.coef_[1]
    
    X_med=list (zip (pred1[med_index],pred2[med_index]))
    Y_med=y_train[med_index]
    model_LR = LinearRegression(fit_intercept=False)
    model_LR.fit(X_med, Y_med)
    med1=model_LR.coef_[0]
    med2=model_LR.coef_[1]
    
    X_high=list (zip (pred1[high_index],pred2[high_index]))
    Y_high=y_train[high_index]
    model_LR = LinearRegression(fit_intercept=False)
    model_LR.fit(X_high, Y_high)
    high1=model_LR.coef_[0]
    high2=model_LR.coef_[1] 
    
    print('The coefficients are:')
    print(low1)
    print(low2)
    print(med1)
    print(med2)
    print(high1)
    print(high2)
    print('The length of three datasets are:')
    print(len(low_index[0]))
    print(len(med_index[0]))
    print(len(high_index[0]))
    print('The length of second training dataset are:')
    print(len(new_train)) 
                     
     
    model_LR = LinearRegression()
    rfe = RFE(model_LR, 5)
    RFEfit = rfe.fit(X_train2, y_train2)
    O3_pred = RFEfit.predict(X_test)
                       
    df2019 = X_test.copy() 
    pred1 = model_RF.predict(np.array(df2019))
    pred2 = model_RF2.predict(np.array(df2019))
    pred3=pred2
    pred3[np.where(pred1<sep1)] = low1*pred1[np.where(pred1<sep1)]+low2*pred2[np.where(pred1<sep1)]
    pred3[np.where((pred1>=sep1)&(pred1<=sep2))] = med1*pred1[np.where((pred1>=sep1)&(pred1<=sep2))]+med2*pred2[np.where((pred1>=sep1)&(pred1<=sep2))]
    pred3[np.where(pred1>sep2)] = high1*pred1[np.where(pred1>sep2)]+high2*pred2[np.where(pred1>sep2)]
    pred3[np.where(pred3<0)] = pred1[np.where(pred3<0)] 
    
    df2019 = dfPm[test_datetime_index].copy() 
    df2019 = df2019.assign(O3_pred=pred3)
    df2019.index = df2019.index + delta
    
    dict_name = str(a)
    dict_of_2019[dict_name] = df2019
    
    temp = df2019.copy()
    o32018 = temp[['O3_pred', 'O3_obs', 'O3_mod']].copy()
    r = pd.date_range(start=o32018.index.min(), end=o32018.index.max(), freq='1H')
    o32018 = o32018.reindex(r)
    o32018['O3_obs'] = o32018['O3_obs'].rolling(8, min_periods=6).mean()
    o32018['O3_mod'] = o32018['O3_mod'].rolling(8, min_periods=6).mean()
    o32018['O3_pred'] = o32018['O3_pred'].rolling(8, min_periods=6).mean()
    o32018['O3avg8hr_org'] = o32018['O3_obs'].shift(-7)
    o32018['O3avg8hr_RF'] = o32018['O3_pred'].shift(-7)
    o32018['O3avg8hr_ap'] = o32018['O3_mod'].shift(-7)
    o32018['O3_obs.maxdaily8hravg'] = o32018['O3avg8hr_org'].rolling(17, min_periods=13).max()
    o32018['O3_pred.maxdaily8hravg'] = o32018['O3avg8hr_RF'].rolling(17, min_periods=13).max() 
    o32018['O3_ap.maxdaily8hravg'] = o32018['O3avg8hr_ap'].rolling(17, min_periods=13).max()
    
    #shift columns
    o32018['O3_obs.maxdaily8hravg'] = o32018['O3_obs.maxdaily8hravg'].shift(-16)
    o32018['O3_pred.maxdaily8hravg'] = o32018['O3_pred.maxdaily8hravg'].shift(-16)
    o32018['O3_ap.maxdaily8hravg'] = o32018['O3_ap.maxdaily8hravg'].shift(-16)
    df2018dailyO38hrmax = o32018[(o32018.index.hour == 7)]#.dropna(how='all')
    
    df2018dailyO38hrmax['AQI_day'] = pd.cut(round(df2018dailyO38hrmax['O3_obs.maxdaily8hravg']),
                            [0, 54, 70, 85, 105, 200, np.inf],
                            labels=[1, 2, 3, 4, 5, 6])
    df2018dailyO38hrmax['AQI_pred_day'] = pd.cut(round(df2018dailyO38hrmax['O3_pred.maxdaily8hravg']),
                            [0, 54, 70, 85, 105, 200, np.inf],
                            labels=[1, 2, 3, 4, 5, 6])
    df2018dailyO38hrmax['AQI_ap_day'] = pd.cut(round(df2018dailyO38hrmax['O3_ap.maxdaily8hravg']),
                            [0, 54, 70, 85, 105, 200, np.inf],
                            labels=[1, 2, 3, 4, 5, 6])

    df2018dailyO38hrmax.index = df2018dailyO38hrmax.index+pd.DateOffset(hours=5)
    
    aqi_low=[0,51,101,151,201,301]
    aqi_high=[50,100,150,200,300,500]
    aqi_lowc=[0,55,71,86,106,201]
    aqi_highc=[54,70,85,105,200,600]
    df2018dailyO38hrmax['AQI']=np.nan
    for i in range(len(df2018dailyO38hrmax)):
        if(np.isnan(df2018dailyO38hrmax['O3_pred.maxdaily8hravg'][i])): continue
        #df2018dailyO38hrmax['AQI'][i] = aqi.to_iaqi(aqi.POLLUTANT_O3_8H, df2018dailyO38hrmax['O3_pred.maxdaily8hravg'][i]/1000, algo=aqi.ALGO_EPA)
        aqi_class=df2018dailyO38hrmax['AQI_pred_day'][i]-1
        df2018dailyO38hrmax['AQI'][i]=(aqi_high[aqi_class]-aqi_low[aqi_class])/(aqi_highc[aqi_class]-aqi_lowc[aqi_class])*(round(df2018dailyO38hrmax['O3_pred.maxdaily8hravg'][i])-aqi_lowc[aqi_class])+aqi_low[aqi_class]
        df2018dailyO38hrmax['AQI'][i]=round(df2018dailyO38hrmax['AQI'][i])
    
    dict_of_max2019[dict_name] = df2018dailyO38hrmax
    a=a+1

with open('D:\Research\ML\cross_validation_aqs\Kennewick_RF2_cv_1719.pkl', 'wb') as fp:
    #with open('/Users/fankai/Research/ML/Kennewick_RF2_cv_1720_noAQI4.pkl',
    #          'wb') as fp:
    pickle.dump(dict_of_2019, fp)

with open('D:\Research\ML\cross_validation_aqs\Kennewick_max_RF2_cv_1719.pkl', 'wb') as fp:
    #with open('/Users/fankai/Research/ML/Kennewick_max_RF2_cv_1720_noAQI4.pkl',
    #          'wb') as fp:
    pickle.dump(dict_of_max2019, fp)