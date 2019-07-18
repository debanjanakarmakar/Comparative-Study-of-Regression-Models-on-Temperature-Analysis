# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:26:53 2019

@author: Prishat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [8,6]

data=pd.read_csv('data.csv')
d=np.array(data)

temp_yearly=d[:,2].reshape([26,12]).sum(axis=1)
temp_yearly=temp_yearly/12
temp_max=d[:,2].reshape([26,12]).max(axis=1)
co2_yearly=np.array([d[i,4] for i in range(len(d)) if i%12==0])
met_yearly=np.array([d[i,5] for i in range(len(d)) if i%12==0])


#plt.scatter(co2_yearly,temp_yearly)
#plt.scatter(met_yearly,temp_yearly)


x=d[:,3:6]
y=d[:,2]

X=x-x.mean(axis=0)
X/=X.std(axis=0)

x_train,x_valid,y_train,y_valid=x[:252],x[252:],y[:252],y[252:]

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

model_1=XGBRegressor(random_state=0)
model_2=XGBRegressor(n_estimators=500,random_state=0)
model_3=XGBRegressor(n_estimators=500,learning_rate=0.1,random_state=0)
model_4=XGBRegressor(n_estimators=500,learning_rate=0.05,random_state=0)
model_5=XGBRegressor(n_estimators=500,learning_rate=0.008,random_state=0)
model_6=XGBRegressor(n_estimators=500,learning_rate=0.05,random_state=0)

model_1.fit(x_train,y_train)
p1=model_1.predict(x_valid)
print("Mean squared Error: " + str(mean_squared_error(p1, y_valid)))

model_2.fit(x_train,y_train)
p2=model_2.predict(x_valid)
print("Mean squared Error: " + str(mean_squared_error(p2, y_valid)))

model_3.fit(x_train,y_train)
p3=model_3.predict(x_valid)
print("Mean squarede Error: " + str(mean_squared_error(p3, y_valid)))

model_4.fit(x_train,y_train)
p4=model_4.predict(x_valid)
print("Mean squared Error: " + str(mean_squared_error(p4, y_valid)))

model_5.fit(x_train,y_train)
p5=model_5.predict(x_valid)
print("Mean squared Error: " + str(mean_squared_error(p5, y_valid)))

model_6.fit(x_train, y_train,early_stopping_rounds=5, eval_set=[(x_valid, y_valid)],verbose=False)
p6=model_6.predict(x_valid)
print("Mean squared Error: " + str(mean_squared_error(p6, y_valid)))
#plt.plot(np.arange(len(y_valid)),y_valid.me,np.arange(len(y_valid)),p6)