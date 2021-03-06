# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:42:38 2020

@author: TQ
"""

from jqdatasdk import *
import pandas as pd
import math
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from pandas.core.frame import DataFrame
'''
auth('*******','********') # account username and password from https://www.joinquant.com/
print(get_query_count())

df=get_price('AU9999.XSGE',
             '2016-01-01', 
         '2020-07-24', 
        frequency='10m', 
          fields=None, 
          skip_paused=False,
          fq='pre', 
         count=None,
         panel=True, 
          fill_paused=True)

df.to_csv('C:\\Users\\TQ\\Desktop\\practice\\201601-202007.csv',index=True)'''
df=pd.read_csv('C:\\Users\\TQ\\Desktop\\practice\\201601-202007.csv',
               parse_dates=True)
df.index=df['time']
df=df.drop('time',axis=1)
df['ln_return']=list(map(lambda x,y:math.log(y/x),df['open'],df['close']))
df['updown']=df['ln_return'].apply(lambda x:1 if x>0 else -1 if x<0 else 0)#updown level

'''
df1=get_price('AU9999.XSGE',
             '2016-01-01', 
         '2020-07-24', 
        frequency='1m', 
          fields=None, 
          skip_paused=False,
          fq='pre', 
         count=None,
         panel=True, 
          fill_paused=True)
df1.to_csv('C:\\Users\\TQ\\Desktop\\pracitce\\minute_data.csv',index=True)'''
df1=pd.read_csv('C:\\Users\\TQ\\Desktop\\pracitce\\minute_data.csv',parse_dates=True)

df1['ln_return']=list(map(lambda x,y:math.log(y/x),df1['open'],df1['close']))


df1['updown']=df1['ln_return'].apply(lambda x:1 if x>0 else -1 if x<0 else 0)


updown=df1['updown'].tolist()
vol=df1['volume'].tolist()
r=df1['ln_return'].tolist()
op=df1['open'].tolist()
corr=[]
acc_vol=[]
acc_updown=[]
for i in range(0,len(updown)+1,10):
    x=sum(updown[i:i+10])
    y=sum(vol[i:i+10])
    z=np.array([op[i:i+10],vol[i:i+10]])
    acc_vol.append(y)
    acc_updown.append(x)
    corr.append(np.corrcoef(z))

fig,axes=plt.subplots(1,1)
sns.distplot(acc_updown,kde_kws={'label':'Distribution of updown'},color='#dc2624')


alpha139=[]#-1*CORR(OPEN,VOLUME,10)
for i in range(0,len(corr)):
    alpha139.append(-1*corr[i][0][1])


df['vwap']=list(map(lambda x,y:(x+y)/2,df['high'],df['low']))
alpha13=list(map(lambda x,y,z:((x*y)**0.5)-z,df['high'],df['low'],df['vwap']))


alpha150=list(map(lambda a,b,c,d:(a+b+c)/(3*d) if d!=0 else 0,df['close'],df['high'],
                  df['low'],df['volume']))


close=df1['close']
alpha34=[]#alpha34???MEAN(CLOSE,10)/CLOSE
mean=[]
for i in range(0,len(close)+1,10):
    m=np.mean(close[i:i+10])
    mean.append(m)
alpha34=list(map(lambda x,y:x/y,mean,df['close']))


data={'ln_return':df['ln_return'],
      'total_volume':acc_vol,
      'level_updown':acc_updown,
      'alpha13':alpha13,
      'alpha34':alpha34,
      'alpha139':alpha139,
      'alpha150':alpha150}
df_data=pd.DataFrame(data,index=df.index)
'''df_data.to_csv('C:\\Users\\TQ\\Desktop\\pracitce\\indicator.csv',index=True,encoding='utf_8_sig')
'''
