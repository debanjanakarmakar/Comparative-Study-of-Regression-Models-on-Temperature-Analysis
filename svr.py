# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:52:40 2019

@author: Debanjana
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

plt.rcParams['figure.figsize']=[8,6]

dataset=pd.read_csv("data.csv")
data=np.array(dataset)

y=data[:,2]
x=data[:,3:6]
x=preprocessing.scale(x)

xtrain,xtest=x[:252],x[252:]
ytrain,ytest=y[:252],y[252:]

emin=100
imin=1
#kernel=poly does not converge so max no. of iterartions have to be fixed
for i in range(1,10):
    regr=SVR(epsilon=0.001,C=2,kernel='poly',degree=i,max_iter=100)
    model=regr.fit(xtrain,ytrain)
    prediction=regr.predict(xtest)
    error=mean_squared_error(ytest,prediction)
    if(emin<error):
        emin=error
        imin=i
print("Error=",emin,"at Gamma=poly and degree=",imin)

regr=SVR(epsilon=0.001,C=20,kernel='sigmoid')
model=regr.fit(xtrain,ytrain)
prediction=regr.predict(xtest)
error=mean_squared_error(ytest,prediction)

#For gamma='rbf' c=20 is the best case with minimum error and
regr =SVR(epsilon=0.001,C=20)
model=regr.fit(xtrain,ytrain)
prediction=regr.predict(xtest)
error=mean_squared_error(ytest,prediction)

plt.figure()
plt.title("SVR")
plt.xlabel("Time in months from 2011")
plt.ylabel("Temperature in Celsius")
plt.plot(np.arange(len(ytest)),ytest,label="Original")
plt.plot(np.arange(len(ytest)),prediction,label="Prediction")
plt.legend()
