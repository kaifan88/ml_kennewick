#pylab
import matplotlib
matplotlib.use('agg')
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
import os
#sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#gaussian_kde
from scipy.stats import gaussian_kde
#anchored text
from matplotlib.offsetbox import AnchoredText 
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from netCDF4 import Dataset
#import aqi
import pickle

exec(open(r"D:/Research/ML/ML_Func_New.py").read())

#idx = int(os.environ['idx'])
'''
# read csv files
#import data from previous 5 years
df = pd.read_csv(r"D:/Research/ML/All_Kennewick_and_Hermiston_O3_met_and_WRF_data_thru_2018.csv")

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

df_right = pd.read_csv('D:/Research/ML/Kennewick_WRF_from_AIRPACT.csv',index_col=0) #Kennewick_full_o3_1720
df_right.index = pd.to_datetime(df_right.index)
#df_right = df_right.iloc[:,list(range(4,9))+[11]]
df_right.columns = [
       'WRF_4km_BCAA.Surface_pres_Pa','WRF_4km_BCAA.PBL_m','WRF_4km_BCAA.T_K',
       'WRF_4km_BCAA.windspeed_m_per_s','WRF_4km_BCAA.WindDir_deg', 
       'WRF_4km_BCAA.RH_pct']
df.update(df_right)

df = df[(df.index.year>2016) & (df.index.month>4) & (df.index.month<10)]

#replace airnow obs with aqs
aqs = pd.read_csv('D:/Research/ML/AQS_O3_Kennewick.csv')
aqs.index = pd.to_datetime(aqs.time)
df = pd.concat([df,aqs],axis=1).drop(columns=['time','Kennewick.O3'])
df = df.rename(columns={"obs": "Kennewick.O3"})

#convert wind direction and spd to U and V components
#df['U_WRF']=-df['WRF_4km_BCAA.windspeed_m_per_s']*np.sin(df['WRF_4km_BCAA.WindDir_deg']*np.pi/180)
#df['V_WRF']=-df['WRF_4km_BCAA.windspeed_m_per_s']*np.cos(df['WRF_4km_BCAA.WindDir_deg']*np.pi/180)
#df['Sea_level_Pressure_mb.Pasco_airport']=df['Sea_level_Pressure_mb.Pasco_airport']*100
#df['Temp_F.Pasco_airport']=(df['Temp_F.Pasco_airport']-32)*5/9+273
'''

df = pd.read_csv('D:/Research/ML/Kennewic_2017_2020.csv')
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']

df['past_O3'] = df['Kennewick.O3'].shift(24)
df['month'] = df.index.month
df['hour'] = df.index.hour
df['Weekday']=df.index.dayofweek

df['Wildfire_smoke_caused_excess_ozone_YN'][df['Wildfire_smoke_caused_excess_ozone_YN']=='Y'] = 1
df['Wildfire_smoke_caused_excess_ozone_YN'][df['Wildfire_smoke_caused_excess_ozone_YN']=='N'] = 0

#exclude the wildfire affected days (only 0.37%)
df=df[(df['Wildfire_smoke_caused_excess_ozone_YN']!=1)]

df_his = df[['Kennewick.O3', 'Kennewick.OZONE_AIRPACT4', 'WRF_4km_BCAA.PBL_m',
                        'WRF_4km_BCAA.Surface_pres_Pa','WRF_4km_BCAA.T_K',
                        'WRF_4km_BCAA.windspeed_m_per_s', 'WRF_4km_BCAA.WindDir_deg', 'WRF_4km_BCAA.RH_pct',
                        'month','hour','Weekday','past_O3'
                        ]].copy()
#rename columns to be more generic
df_his.columns = ['O3_obs', 'O3_ap', 'PBL_m', 'Surface_Pressure_Pa', 'Temp_K', 'Wind_speed_m_per_s', 'Wind_direction_deg',
                    'RH_pct','Month','Hour','Weekday' , 'Past_O3']

df_his.loc[df_his.index.year > 2018, ['PBL_m', 'Surface_Pressure_Pa', 'Temp_K', 'Wind_speed_m_per_s', 'Wind_direction_deg', 'RH_pct']] = np.nan

'''
#read new obs data
#obs_ap=pd.read_csv(r"http://lar.wsu.edu/R_apps/2020ap5/data/byAQSID/530050003.apan")
obs_ap = pd.read_csv('D:/Research/ML/530050003_2020.apan')
obs_ap['DateTime'] = pd.to_datetime(obs_ap['DateTime'])
obs_ap.index = obs_ap['DateTime']
delta = np.timedelta64(8,'h')
obs_ap.index = obs_ap.index - delta
#example
#obs_ap.iloc[1992:2064,-8]=df['O3_obs'][8431:8503].values
r = pd.date_range(start=obs_ap.index.min(), end=obs_ap.index.max(), freq='1H')
obs_ap = obs_ap.reindex(r)

obs_ap['O3_obs']=obs_ap['OZONEan'].copy()
obs_ap['O3_ap']=obs_ap['OZONEap'].copy()
obs_ap['Past_O3'] = obs_ap['O3_obs'].shift(24)
#obs_ap = obs_ap[['Past_O3','O3_obs','O3_ap']]
obs_ap = obs_ap[['O3_ap']]
#df.update(obs_ap.rename(columns={'OZONEan': 'O3_obs'}))
#df = df.append(obs_ap.rename(columns={'O3avg8hr': 'O3_obs','O3avg8hr_ap': 'O3_ap'}))
#obs_ap = obs_ap.rename(columns={'O3avg8hr': 'O3_obs','O3avg8hr_ap': 'O3_ap'})

#f = open("/data/lar/users/kfan/ml_project/data.pkl",'rb')
#dict_of_dt = pickle.load(f)
'''

t2 = pd.read_csv('D:/Research/ML/wrf/t2.csv')
t2.index = pd.to_datetime(t2['date'])
del t2['date']
rh2 = pd.read_csv('D:/Research/ML/wrf/rh2.csv')
rh2.index = pd.to_datetime(rh2['date'])
del rh2['date']
pblh = pd.read_csv('D:/Research/ML/wrf/pblh.csv')
pblh.index = pd.to_datetime(pblh['date'])
del pblh['date']

sfp = pd.read_csv('D:/Research/ML/wrf/sfp.csv')
sfp.index = pd.to_datetime(sfp['date'])
#slp1 = pd.read_csv('/home/airpact5/ECY/ML_4cast/WRF_4km_surface_met_at_Kennewick_AQSID.530050003.csv')
#slp1.index = slp1['date']
#slp1 = slp.merge(slp1, how='left')
#for i in slp.keys():
#    slp[i] = slp1['WRF_4km.Kennewick_AQSID.530050003.Surface_pres_Pa'].values
del sfp['date']
#sfp = slp*np.exp(-0.1963/0.029/t2)*100 #-270
#sfp = slp *100 - 2454
'''
sfp = pd.read_csv('D:/Research/ML/wrf/sfp.csv')
sfp.index = sfp['date']
del sfp['date']
'''
spd10 = pd.read_csv('D:/Research/ML/wrf/spd10.csv')
spd10.index = pd.to_datetime(spd10['date'])
del spd10['date']
dir10 = pd.read_csv('D:/Research/ML/wrf/dir10.csv')
dir10.index = pd.to_datetime(dir10['date'])
del dir10['date']
#dir10 = dir10 - 90
#dir10 %= 360 

for idx in range(16,27):
    print(idx)
    wrf_members = [t2.keys()[idx]] #list(t2.keys())
    wrf_var = ['t2','rh2','spd10','dir10','pblh','sfp']
    wrf_var_name = ['Temp_K','RH_pct','Wind_speed_m_per_s', 'Wind_direction_deg','PBL_m','Surface_Pressure_Pa']
    
    dict_of_dt=dict()
    dict_of_dfs=dict()
    RF1_feature_table = pd.DataFrame(columns = df_his.keys()[2:]) 
    RF2_feature_table = pd.DataFrame(columns = df_his.keys()[2:])
    
    m = 0
    #for m in range(15):
    print(wrf_members)
    df_tmp = df_his.copy()
    
    '''
    df_tmp = df_tmp.combine_first(obs_ap)
    df_tmp.update(obs_ap)
    '''
    
    for i in range(0,6):
        df_tmp = df_tmp.combine_first(eval(wrf_var[i])[[wrf_members[m]]].rename(columns={wrf_members[m]: wrf_var_name[i]}))
        df_tmp.update(eval(wrf_var[i])[[wrf_members[m]]].rename(columns={wrf_members[m]: wrf_var_name[i]}))
    
    df_tmp['Weekday']=df_tmp.index.dayofweek
    df_tmp['Month'] = df_tmp.index.month
    df_tmp['Hour'] = df_tmp.index.hour
    
    #preprocessing
    dfPm = df_tmp[['O3_obs', 'O3_ap', 'PBL_m', 'Surface_Pressure_Pa', 'Temp_K', 'Wind_speed_m_per_s', 'Wind_direction_deg',
                    'RH_pct', 'Past_O3','Month','Hour','Weekday']].dropna().copy()
    #dfPm = dfPm[(dfPm.index<'2018-08-16 07:00:00') | (dfPm.index>'2018-08-17 06:00:00')] #remove AQI 4 day
    X_org = dfPm.drop(['O3_obs', 'O3_ap'], 1).copy()
    X_dat = pd.DataFrame(preprocess('MAS', X_org))
    X_dat.columns = X_org.keys()
    X_dat.index = X_org.index
    
    #training RF and MLR model
    #RF_classifier
    df2019_org = dfPm[(dfPm.index.year > 2018)].copy()
    df2019_org['O3_pred'] = np.nan
    for d in sorted(set(dfPm[(dfPm.index.year > 2018)].index.date)):
        print(d)
        if(len(X_dat[(X_dat.index.date == d)])==0): continue
        
        #RF
        df1217 = dfPm[(dfPm.index.date < d)].copy()
        X = np.array(X_dat[(X_dat.index.date < d)])
        # separate "label" to y 
        Y = np.array(df1217['O3_obs'])
            
        X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=42)
        
        
        # feature extraction
        model_RF = RandomForestRegressor(n_estimators=100, max_depth=7,random_state=137)
        model_RF = model_RF.fit(X, Y)
        
        new_train = pd.DataFrame(X_train)
        new_train['obs_o3'] = Y_train
        new_train['pred_o3'] = model_RF.predict(X_train)
        new_train['diff_o3'] = abs(new_train['pred_o3']-new_train['obs_o3'])
        new_train=new_train[new_train['diff_o3']>5]
        X_train2 = np.array(new_train.drop(['obs_o3','pred_o3','diff_o3'],axis=1))
        Y_train2 = Y_train[new_train.index]
        model_RF2 = RandomForestRegressor(n_estimators=200, max_depth=7,random_state=137)
        model_RF2 = model_RF2.fit(X_train2, Y_train2)
        
        RF1_feature_table = RF1_feature_table.append(dict(zip(RF1_feature_table.columns, model_RF.feature_importances_)), ignore_index=True)
        RF2_feature_table = RF2_feature_table.append(dict(zip(RF2_feature_table.columns, model_RF2.feature_importances_)), ignore_index=True)
        RF1_feature_table.to_csv('D:/Research/ML/ensemble/'+wrf_members[m]+'ML2_time_RF1_feature.csv',index=False)
        RF2_feature_table.to_csv('D:/Research/ML/ensemble/'+wrf_members[m]+'ML2_time_RF2_feature.csv',index=False)
        
        from prettytable import PrettyTable
        from prettytable import MSWORD_FRIENDLY
        tab = PrettyTable()
        tab.add_column("feature_sel", X_dat.keys() )
        tab.add_column("RF1",["%.3f" % member for member in model_RF.feature_importances_ ]) 
        tab.add_column("RF2",["%.3f" % member for member in model_RF2.feature_importances_ ])   
        tab.set_style(MSWORD_FRIENDLY)
        print(tab)
        
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
        Y_low=Y_train[low_index]
        model_LR = LinearRegression(fit_intercept=False)
        model_LR.fit(X_low, Y_low)
        low1=model_LR.coef_[0]
        low2=model_LR.coef_[1]
        
        X_med=list (zip (pred1[med_index],pred2[med_index]))
        Y_med=Y_train[med_index]
        model_LR = LinearRegression(fit_intercept=False)
        model_LR.fit(X_med, Y_med)
        med1=model_LR.coef_[0]
        med2=model_LR.coef_[1]
        
        X_high=list (zip (pred1[high_index],pred2[high_index]))
        Y_high=Y_train[high_index]
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
                                   
        df2019 = X_dat[(X_dat.index.date == d)].copy() #2015)] #
        
        pred1 = model_RF.predict(np.array(df2019))
        pred2 = model_RF2.predict(np.array(df2019))
        pred3=pred2
        pred3[np.where(pred1<sep1)] = low1*pred1[np.where(pred1<sep1)]+low2*pred2[np.where(pred1<sep1)]
        pred3[np.where((pred1>=sep1)&(pred1<=sep2))] = med1*pred1[np.where((pred1>=sep1)&(pred1<=sep2))]+med2*pred2[np.where((pred1>=sep1)&(pred1<=sep2))]
        pred3[np.where(pred1>sep2)] = high1*pred1[np.where(pred1>sep2)]+high2*pred2[np.where(pred1>sep2)]
        pred3[np.where(pred3<0)] = pred1[np.where(pred3<0)] 
        
        df2019 = dfPm[(dfPm.index.date == d)].copy()
        df2019['O3_pred'] = pred3
        
        tmp=pd.DataFrame(data=df2019['O3_pred'],index=df2019.index,columns=['O3_pred'])
        df2019_org.update(tmp)
    
    dict_of_dt[wrf_members[m]] = df2019_org
    
    temp = df2019_org.copy()
    o32018 = temp[['O3_pred', 'O3_obs', 'O3_ap']].copy()
    r = pd.date_range(start=o32018.index.min(), end=o32018.index.max(), freq='1H')
    o32018 = o32018.reindex(r)
    o32018['O3_obs'] = o32018['O3_obs'].rolling(8, min_periods=6).mean()
    o32018['O3_ap'] = o32018['O3_ap'].rolling(8, min_periods=6).mean()
    o32018['O3_pred'] = o32018['O3_pred'].rolling(8, min_periods=6).mean()
    o32018['O3avg8hr_org'] = o32018['O3_obs'].shift(-7)
    o32018['O3avg8hr_RF'] = o32018['O3_pred'].shift(-7)
    o32018['O3avg8hr_ap'] = o32018['O3_ap'].shift(-7)
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
    
    aqi_low=[0,51,101,151,201]
    aqi_high=[50,100,150,200,300]
    aqi_lowc=[0,55,71,86,106]
    aqi_highc=[54,70,85,105,200]
    df2018dailyO38hrmax['AQI']=np.nan
    for i in range(len(df2018dailyO38hrmax)):
        if(np.isnan(df2018dailyO38hrmax['O3_pred.maxdaily8hravg'][i])): continue
        #df2018dailyO38hrmax['AQI'][i] = aqi.to_iaqi(aqi.POLLUTANT_O3_8H, df2018dailyO38hrmax['O3_pred.maxdaily8hravg'][i]/1000, algo=aqi.ALGO_EPA)
        aqi_class=df2018dailyO38hrmax['AQI_pred_day'][i]-1
        df2018dailyO38hrmax['AQI'][i]=(aqi_high[aqi_class]-aqi_low[aqi_class])/(aqi_highc[aqi_class]-aqi_lowc[aqi_class])*(round(df2018dailyO38hrmax['O3_pred.maxdaily8hravg'][i])-aqi_lowc[aqi_class])+aqi_low[aqi_class]
        df2018dailyO38hrmax['AQI'][i]=round(df2018dailyO38hrmax['AQI'][i])
        
    dict_of_dfs[wrf_members[m]] = df2018dailyO38hrmax
    
    df2018dailyO38hrmax.to_csv('D:/Research/ML/ensemble/max8hr_twoRF_'+wrf_members[m]+'_right.csv',index=True)

#with open('/data/lar/users/kfan/ml_project/hourly_2019_twoRF_right1.pkl', 'wb') as fp:
#    pickle.dump(dict_of_dt, fp)

#with open('/data/lar/users/kfan/ml_project/max8hr_2019_twoRF_right1.pkl', 'wb') as fp:
#    pickle.dump(dict_of_dfs, fp)

'''
f = open("/data/lar/users/kfan/ml_project/max8hr_2019_twoRF.pkl",'rb')
dict_of_dfs = pickle.load(f)

for m in range(len(wrf_members)):
    dict_of_dfs[wrf_members[m]] =  dict_of_dfs[wrf_members[m]][dict_of_dfs[wrf_members[m]].index<'2019-06-05 12:00:00']

for m in range(len(wrf_members)):
    r = pd.date_range(start=dict_of_dfs[wrf_members[m]].index.min(), end=dict_of_dfs[wrf_members[m]].index.max())
    dict_of_dfs[wrf_members[m]] = dict_of_dfs[wrf_members[m]].reindex(r)
'''
'''
sample = dict_of_dfs['mean']
sample['O3_pred.maxdaily8hravg'] = pd.concat(dict_of_dfs).groupby(level=1).mean()['O3_pred.maxdaily8hravg']
sample['AQI_pred_day'] = pd.cut(round(sample['O3_pred.maxdaily8hravg']),
                                [0, 54, 70, 85, 105, 200, np.inf],
                                labels=[1, 2, 3, 4, 5, 6])
sample['color'] = pd.cut(sample['AQI_pred_day'],
                            [0, 1, 2, 3, 4, 5, 6],
                            labels=['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023'])
sample = sample.rename(columns={'O3_obs.maxdaily8hravg': 'Measured', 'O3_pred.maxdaily8hravg': 'Model', 'O3_ap.maxdaily8hravg': 'AIRPACT'})

diff = 0
sumo = 0
for m in range(len(wrf_members)):
    diff = diff + (abs(dict_of_dfs[wrf_members[m]]['O3_pred.maxdaily8hravg']-sample['Model'])).sum()
    sumo = sumo + (sample['Model']).sum()

uncertainty = round(diff/sumo*100,2)
print('uncertainty =',uncertainty,'%')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#colors = plt.cm.rainbow(np.linspace(0, 1, (len(wrf_members)-2)))
for m in range(len(wrf_members)):
    if wrf_members[m] in ['mean']: continue
    else: line4=plt.plot(dict_of_dfs[wrf_members[m]].index,dict_of_dfs[wrf_members[m]]['O3_pred.maxdaily8hravg'],color='grey',alpha=0.3, label='Uncertainty')

line1=plt.plot(sample.index,sample['Model'],label='Forecast',color='blue')
line2=plt.plot(sample.index,sample['Measured'],'black')
#line3=plt.plot(sample.index,sample['AIRPACT'],'red')
#line5=plt.plot(dict_of_dfs['WRF'].index,dict_of_dfs['WRF']['O3_pred.maxdaily8hravg'],label='WRF-GFS',color='orange')
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
leg = ax.legend(handles, labels, loc=2, prop={'size': 10})
ax.add_artist(leg)
#ax.set_xticklabels(sample.index.date, rotation=20)
import matplotlib.dates as mdates
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Daily maximum 8-hr averaged O3 concentration (ppb)', fontsize=10)
ax.axhspan(0, 54.5, facecolor='#00E400')
ax.axhspan(54.5, 70.5, facecolor='#FFFF00')
ax.axhspan(70.5, 85.5, facecolor='#FF7E00')
if max(sample['AQI'][4:8])>150: ax.axhspan(85.5, 105.5, facecolor='#FF0000')
elif max(sample['AQI'][4:8])>200: ax.axhspan(100.5, 200.5, facecolor='#8F3F97')
else: pass

import matplotlib.patches as mpatches
patch6 = mpatches.Patch(color='#00E400', label='Good')
patch5 = mpatches.Patch(color='#FFFF00', label='Moderate')
patch4 = mpatches.Patch(color='#FF7E00', label='Unhealthy for Sensitive Groups')
patch3 = mpatches.Patch(color='#FF0000', label='Unhealthy')
patch2 = mpatches.Patch(color='#8F3F97', label='Very Unhealthy')
#patch1 = mpatches.Patch(color='red', label='Unhealthy')
if max(sample['AQI'][4:8])>200: all_handles = (patch2, patch3, patch4, patch5, patch6)
else: all_handles = (patch3, patch4, patch5, patch6)

leg = ax.legend(loc=4,handles=all_handles)
ax.add_artist(leg)
if max(sample['AQI'][4:8])>150: ax.set_ylim([0, 105])
elif max(sample['AQI'][4:8])>200: ax.set_ylim([0, 200])
else: ax.set_ylim([0, 85])

plt.savefig('/data/lar/users/kfan/ml_project/line_all_twoRF_right1.png',dpi=1000)


fig = plt.figure(figsize=(24,4))
sample = sample.dropna()
DenGraphWithStats(fig, sample['Measured'].values, sample['AIRPACT'].values, "AIRPACT", 1)
q = np.linspace(54, 54, 100)
j = np.linspace(0, 100, 100)
plt.plot(q,j, 'b')
plt.plot(j,q, 'b')

DenGraphWithStats(fig, sample['Measured'].values, sample['Model'].values, "Two-phase RF", 2)
q = np.linspace(54, 54, 100)
j = np.linspace(0, 100, 100)
plt.plot(q,j, 'b')
plt.plot(j,q, 'b')

plt.tight_layout()
plt.savefig('/data/lar/users/kfan/ml_project/scatter_all_twoRF_right1.png',dpi=1000)

PT_model_comp_df('Measured', 'Model', sample)

r2(sample,'Measured','Model')
'''
