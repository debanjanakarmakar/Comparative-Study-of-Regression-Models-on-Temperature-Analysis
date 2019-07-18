
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("data.csv")
data=np.array(dataset)

plt.rcParams['figure.figsize']=[8,6]

y=data[:,2]
x=data[:,3:6]


xtrain,xtest=x[:252],x[252:]
ytrain,ytest=y[:252],y[252:]
emin=100
imin=1

for i in range(1,100):
    regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=i,criterion="mse")
    regr.fit(xtrain,ytrain)#Gives minimum error in n_estimators=42
    prediction=regr.predict(xtest)
    error=mean_squared_error(ytest,prediction)
    if(emin>error):
        emin=error
        imin=i
        
print("Error",error,"  at max_depth of tree= ",imin)
    
emin=100
imin=1
for i in range(1,100):
    regr = RandomForestRegressor(max_depth=i, random_state=0, n_estimators=42,criterion="mse")
    regr.fit(xtrain,ytrain)#Gives minimum error in n_estimators=42
    prediction=regr.predict(xtest)
    error=mean_squared_error(ytest,prediction)
    if(emin>error):
        emin=error
        imin=i
        
print("Error",error,"  at n_estimators= ",imin)

regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=42,criterion="mse")
regr.fit(xtrain,ytrain)#Gives minimum error in n_estimators=42
prediction=regr.predict(xtest)
error=mean_squared_error(ytest,prediction)
print("Mean Squared Error=",error)
plt.figure()
plt.title("Random Forest")
plt.xlabel("Time in months from 2011")
plt.ylabel("Temperature in Celsius")
plt.plot(np.arange(len(ytest)),ytest,label="Original")
plt.plot(np.arange(len(ytest)),prediction,label="Prediction")
plt.legend()