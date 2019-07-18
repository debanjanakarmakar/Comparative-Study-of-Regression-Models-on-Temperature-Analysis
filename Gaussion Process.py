# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:51:33 2019

@author: Prishat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [8,6]

#Reading the Data
data=pd.read_csv('data.csv')
d=np.array(data)

#creating the yearly values
temp_yearly=d[:,2].reshape([26,12]).sum(axis=1)
temp_yearly=temp_yearly/12
temp_max=d[:,2].reshape([26,12]).max(axis=1)
co2_yearly=np.array([d[i,4] for i in range(len(d)) if i%12==0])
met_yearly=np.array([d[i,5] for i in range(len(d)) if i%12==0])


#plt.scatter(co2_yearly,temp_yearly)
#plt.scatter(met_yearly,temp_yearly)

#Initializing x and y
x=d[:,3:6]
y=d[:,2]

#Normalizing the features
X=x-x.mean(axis=0)
X/=X.std(axis=0)

#splitting the data into test
x_train,x_valid,y_train,y_valid=x[:252],x[252:],y[:252],y[252:]


#building the model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_squared_error

kernel = DotProduct() + WhiteKernel()
kernel_2 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-1, 1e-1))

gp = GaussianProcessRegressor(kernel=kernel,alpha=2,n_restarts_optimizer=5,random_state=0,normalize_y=True)

gp.fit(x_train[:,0].reshape((-1,1)), y_train)

y_pred, sigma = gp.predict(x_valid[:,0].reshape((-1,1)) ,return_std=True)
print("Mean squared Error: " ,mean_squared_error(y_pred, y_valid))
#print(gp.score(x_valid,y_valid))


#Plotting the graph with confidence interval
plt.figure()
plt.title("XGBoost Regression")
plt.plot(np.arange(len(y_valid)), y_valid, 'r:', label='original')
plt.plot(np.arange(len(y_valid)), y_pred, label='Prediction')
x_v=np.arange(len(y_valid))
plt.fill(np.concatenate([x_v, x_v[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('Time (in months from 2011)')
plt.ylabel('Temperature(in degree Celsius)')
plt.legend()
