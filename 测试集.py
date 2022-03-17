# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:55:15 2020

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
from hmmlearn.hmm import GaussianHMM
import datetime
auth('18665883365','Hu12345678')
print(get_query_count())

df=get_price('AU9999.XSGE',
             '2019-08-19',
         '2020-08-21', 
        frequency='10m', 
          fields=None, 
          skip_paused=False,
          fq='pre', 
         count=None,
         panel=True, 
          fill_paused=True)




 
df['对数收益率']=list(map(lambda x,y:math.log(y/x),df['open'],df['close']))

df['updown']=df['对数收益率'].apply(lambda x:1 if x>0 else -1 if x<0 else 0)#计算正负强度


df1=get_price('AU9999.XSGE',
             '2019-08-19',
         '2020-08-21',     
        frequency='1m', 
          fields=None, 
          skip_paused=False,
          fq='pre', 
         count=None,
         panel=True, 
          fill_paused=True)

df1['对数收益率']=list(map(lambda x,y:math.log(y/x),df1['open'],df1['close']))


df1['updown']=df1['对数收益率'].apply(lambda x:1 if x>0 else -1 if x<0 else 0)


updown=df1['updown'].tolist()
vol=df1['volume'].tolist()
r=df1['对数收益率'].tolist()
op=df1['open'].tolist()
corr=[]
acc_vol=[]
acc_updown=[]
for i in range(0,len(updown)+1,10):
    x=sum(updown[i:i+10])
    y=sum(vol[i:i+10])
    z=np.array([op[i:i+10],vol[i:i+10]])
    acc_vol.append(y)#累计交易量
    acc_updown.append(x)#正负强度
    corr.append(np.corrcoef(z))



alpha139=[]#计算alpha139：-1*CORR(OPEN,VOLUME,10)
for i in range(0,len(corr)):
    alpha139.append(-1*corr[i][0][1])


df['vwap']=list(map(lambda x,y:(x+y)/2,df['high'],df['low']))
alpha13=list(map(lambda x,y,z:((x*y)**0.5)-z,df['high'],df['low'],df['vwap']))
#计算alpha13

alpha150=list(map(lambda a,b,c,d:(a+b+c)/(3*d) if d!=0 else 0,df['close'],df['high'],
                  df['low'],df['volume']))
#计算alpha150

close=df1['close']
alpha34=[]#计算alpha34：MEAN(CLOSE,10)/CLOSE
mean=[]
for i in range(0,len(close)+1,10):
    m=np.mean(close[i:i+10])
    mean.append(m)
alpha34=list(map(lambda x,y:x/y,mean,df['close']))

data={'对数收益率':df['对数收益率'],
      '累计交易量':acc_vol,
      '累计正负强度':acc_updown,
      'alpha13':alpha13,
      'alpha34':alpha34,
      'alpha139':alpha139,
      'alpha150':alpha150}
df=pd.DataFrame(data,index=df.index)





df=df.apply(lambda x:(x-np.mean(x))/np.std(x))#标准化处理



df['alpha139'].fillna(0, inplace=True)
for i in range(0,7):
    per_1=np.percentile(df.iloc[:,i],1)
    per_99=np.percentile(df.iloc[:,i],99)
    
    df.iloc[:,i]=df.iloc[:,i].apply(lambda x: per_1
                                 if x<per_1
                                 else per_99
                                 if x>per_99 
                                 else x)
#-----------------------------------------------
df_test=df


df=pd.read_csv('C:\\Users\\TQ\\Desktop\\实习\\处理后的数据.csv',
               parse_dates=True)

lnr=list(map(float,df['对数收益率']))

alpha13=list(map(float,df['alpha13']))
alpha34=list(map(float,df['alpha34']))
alpha139=list(map(float,df['alpha139']))
alpha150=list(map(float,df['alpha150']))
vol=df['累计交易量']
vol=list(map(float,vol))
updown=df['累计正负强度']
updown=list(map(float,updown))

A=np.column_stack([lnr,alpha13,alpha34,alpha139,alpha150,vol,updown])
model=GaussianHMM(n_components=4,n_iter=500)
model.fit(A)
states=model.predict(A)


lnr=list(map(float,df_test['对数收益率']))

alpha13=list(map(float,df_test['alpha13']))
alpha34=list(map(float,df_test['alpha34']))
alpha139=list(map(float,df_test['alpha139']))
alpha150=list(map(float,df_test['alpha150']))
vol=df_test['累计交易量']
vol=list(map(float,vol))
updown=df_test['累计正负强度']
updown=list(map(float,updown))

B=np.column_stack([lnr,alpha13,alpha34,alpha139,alpha150,vol,updown])


states_test=model.predict(B)



df_test['类别']=states_test.tolist()
classinformation=df_test['类别'].unique()

for temp_classinformation in classinformation:
    temp_data=df_test[df_test['类别'].isin([temp_classinformation])]
    exec("df%s=temp_data"%temp_classinformation)

dic={'df0':df0['对数收益率'].mean(),
     'df1':df1['对数收益率'].mean(),
     'df2':df2['对数收益率'].mean(),
     'df3':df3['对数收益率'].mean(),}
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

    
df=get_price('AU9999.XSGE',
             '2019-08-19',
         '2020-08-21', 
        frequency='10m', 
          fields=None, 
          skip_paused=False,
          fq='pre', 
         count=None,
         panel=True, 
          fill_paused=True)
close=df['close'].values
date=pd.to_datetime(df.index)
open=df['open'].values
time_up=[]
r_up=[]
count_up=0
i=0
j=0
df['time']=pd.to_datetime(df.index)
df['date']=df['time'].dt.hour
Time=df['date'].tolist()
print('开始计算')
while i<len(close)-1:
    if(states[i]==max):
        buy=i

        for j in range(0,len(close)-1):
            if i+j>len(close)-1:
                break
            if states_test[i+j]!=max:
                sell=j+i
                r1=((close[sell]-open[buy])/open[buy])-0.0002
                r_up.append(r1)
                i=sell+1
                count_up=count_up+1

                

                if Time[sell]-Time[buy]>8:
                    time_up.append(date[sell]-date[buy])
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

        for j in range(0,len(close)-1):
            if states_test[i+j]!=min:
                buy=j+i
                r1=((close[sell]-open[buy])/close[sell])-0.0002
                r_down.append(r1)
                i=buy+1
                count_down=count_down+1
                

                if Time[buy]-Time[sell]>8:
                   time_down.append(date[buy]-date[sell])
                if Time[buy]-Time[sell]<2:
                   time_down.append(date[buy]-date[sell])
                elif Time[buy]-Time[sell]>=6:
                    time_down.append(date[buy]-date[sell]-datetime.timedelta(hours=6))
                elif Time[buy]-Time[sell]>=2 &Time[buy]-Time[sell]<6:
                    time_down.append(date[buy]-date[sell]-datetime.timedelta(hours=2))
                break
    else: 
        i=i+1


print('做多的总收益为：',np.sum(r_up))
print('做多的次数为：',count_up)
print('做多的平均持仓时间：',np.mean(time_up))
print('做空的总收益为：',np.sum(r_down))
print('做空的次数为：',count_down)
print('做空的平均持仓时间：',np.mean(time_down))

data={'做多的总收益':np.sum(r_up),
      '做多的平均收益':np.mean(r_up),
      '做多的交易次数':count_up,
      '做多的平均持仓时间':np.mean(time_up),
      '做空的总收益':np.sum(r_down),
      '做空的平均收益':np.mean(r_down),
      '做空的交易次数':count_down,
      '做空的平均持仓时间':np.mean(time_down)}
symbol=['策略']
df_r=pd.DataFrame(data,index=symbol)
df_r.to_excel('C:\\Users\\TQ\\Desktop\\实习\\测试集收益结果2019.xlsx',index=True,encoding='utf_8_sig')

'''df_date=get_price('AU9999.XSGE',
             '2019-08-19',
         '2020-08-21', 
        frequency='10m', 
          fields=None, 
          skip_paused=False,
          fq='pre', 
         count=None,
         panel=True, 
          fill_paused=True)
close=df_date['close'].values
date=pd.to_datetime(df_date.index)

open=df_date['open'].values

plt.figure(figsize=(16, 6),dpi=100) 
 
for j in range(len(close)-1):
        if states_test[j] == max:
            plt.plot([date[j],date[j+1]],[open[j],close[j]],color = 'r')
        if states_test[j] == min:
            plt.plot([date[j],date[j+1]],[open[j],close[j]],color = 'b')
        if states_test[j] == max2:
            plt.plot([date[j],date[j+1]],[open[j],close[j]],color = 'g')
        if states_test[j] == min2:
            plt.plot([date[j],date[j+1]],[open[j],close[j]],color = 'orange')

plt.show()'''