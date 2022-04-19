# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:09:42 2020

@author: TQ
"""

import pandas as pd
from jqdatasdk import *
import math
import numpy as np
import datetime
from matplotlib import pyplot as plt
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from time import *
from pandas.plotting import register_matplotlib_converters
from pandas.core.frame import DataFrame

df=pd.read_csv('C:\\Users\\TQ\\Desktop\\实习\\Data after Standardlization.csv',
               parse_dates=True)

lnr=list(map(float,df['ln_return']))

alpha13=list(map(float,df['alpha13']))
alpha34=list(map(float,df['alpha34']))
alpha139=list(map(float,df['alpha139']))
alpha150=list(map(float,df['alpha150']))
vol=df['Total_volume']
vol=list(map(float,vol))
updown=df['level_updown']
updown=list(map(float,updown))

A=np.column_stack([lnr,alpha13,alpha34,alpha139,alpha150,vol,updown])
model=GaussianHMM(n_components=4,n_iter=500)
model.fit(A)
states=model.predict(A)

prob=model.predict_proba(A)

print(states)



df['type']=states.tolist()
classinformation=df['type'].unique()

for temp_classinformation in classinformation:
    temp_data=df[df['type'].isin([temp_classinformation])]
    exec("df%s=temp_data"%temp_classinformation)

dic={'df0':df0['ln_return'].mean(),
     'df1':df1['ln_return'].mean(),
     'df2':df2['ln_return'].mean(),
     'df3':df3['ln_return'].mean(),}
d_order=sorted(dic.items(),key=lambda x:x[1],reverse=False) 
max = sorted(dic, key=lambda x: dic[x],reverse=True)[0]
max=int(max[2])
print(max)
min = sorted(dic, key=lambda x: dic[x],reverse=True)[3]
min=int(min[2])

max2=sorted(dic, key=lambda x: dic[x],reverse=True)[1]
min2=sorted(dic, key=lambda x: dic[x],reverse=True)[2]
max2=int(max2[2])
min2=int(min2[2])
'''
writer=pd.ExcelWriter('C:\\Users\\TQ\\Desktop\\practice\\hmm.xlsx')
df0.mean().to_excel(writer,)
df1.mean().to_excel(writer,startcol=8)
df2.mean().to_excel(writer,startrow=10)
df3.mean().to_excel(writer,startcol=8,startrow=10)
writer.save()'''

df=pd.read_csv('C:\\Users\\TQ\\Desktop\\practice\\201601-202007.csv',
               parse_dates=True)
close=df['close'].values
date=pd.to_datetime(df['time'])
open=df['open'].values
time_up=[]
r_up=[]
count_up=0
i=0
j=0
df['time']=pd.to_datetime(df['time'])
df['date']=df['time'].dt.hour
Time=df['date'].tolist()
while i<len(close):
    if(states[i]==max):
        buy=i

        for j in range(0,len(close)):
            if states[i+j]!=max:
                sell=j+i
                r1=((close[sell]-open[buy])/open[buy])-0.0002
                r_up.append(r1)
                i=sell+1
                count_up=count_up+1
                if Time[sell]-Time[buy]<2:
                    time_up.append(date[sell]-date[buy])
                elif Time[sell]-Time[buy]>=6:
                    time_up.append(date[sell]-date[buy]-datetime.timedelta(hours=6))
                elif Time[sell]-Time[buy]>=2 & Time[sell]-Time[buy]<6:
                    time_up.append(date[sell]-date[buy]-datetime.timedelta(hours=2))
                break
    else: 
        i=i+1
i=0
j=0
time_down=[]
r_down=[]
count_down=0
while i<len(close):
    if(states[i]==min):
        sell=i

        for j in range(0,len(close)):
            if states[i+j]!=min:
                buy=j+i
                r1=((close[sell]-open[buy])/close[sell])-0.0002
                r_down.append(r1)
                i=buy+1
                count_down=count_down+1
                if Time[buy]-Time[sell]<2:
                   time_down.append(date[buy]-date[sell])
                elif Time[buy]-Time[sell]>=6:
                    time_down.append(date[buy]-date[sell]-datetime.timedelta(hours=6))
                elif Time[buy]-Time[sell]>=2 &Time[buy]-Time[sell]<6:
                    time_down.append(date[buy]-date[sell]-datetime.timedelta(hours=2))
                break
    else: 
        i=i+1


print('Total profit from buying long：',np.sum(r_up))
print('Total count for buying long：',count_up)
print('Average duration of buying long positions ：',np.mean(time_up))
print('Total profit from selling short：',np.sum(r_down))
print('Total count for selling short：',count_down)
print('Average duration of selling short positions：',np.mean(time_down))

'''
data={'Total profit from buying long':np.sum(r_up),
      'Average profit from buying long':np.mean(r_up),
      'Total count for buying long:':count_up,
      'Average duration of buying long positions':np.mean(time_up),
      'Total profit from selling short':np.sum(r_down),
      'Average from selling short':np.mean(r_down),
      'Total count for selling short':count_down,
      'Average duration of selling short positions':np.mean(time_down)}
symbol=['strategy']
df_r=pd.DataFrame(data,index=symbol)
