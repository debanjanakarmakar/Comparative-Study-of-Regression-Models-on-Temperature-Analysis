
# Comparative-Study-of-Regression-Models-on-Temperature-Analysis
Regression models used

1.SVR 

2.Random Forest

3.Gaussian Process

4.XGBoost

## Support Vector Regression

Support Vector Regression is the model that when run on the dataset, gives best results when kernel used is Radial Basis Function.

![svr](https://user-images.githubusercontent.com/43705726/61465067-01630a80-a995-11e9-9724-add7b029b8d9.png)


The minimum mean sqared error= 10.435019277416062

The parameters used are:
kernel=rbf by default
epsilon=0.001
C=20

## Random Forest

Random Forest is the best regression model for predictive analysis.

![randomforest](https://user-images.githubusercontent.com/43705726/61465066-00ca7400-a995-11e9-8869-1e3d3f0b438f.png)

Parameters for best result:
max_depth=2

random_state=0

n_estimators=42

criterion="mse"

Minimum Mean squared error =6.301781523292209

## XGBoost Regression:

![XGB](https://user-images.githubusercontent.com/37043631/61465058-fd36ed00-a994-11e9-9ba7-3bae23e35d6b.png)

Mean Squared Error: 6.827033810901381

### Parameters:-

n_estimators= 500

learning_rate= 0.05

early_stopping_rounds=5

## Gaussian Process Regression:

![GP](https://user-images.githubusercontent.com/37043631/61465094-0922af00-a995-11e9-9337-6b85be4cdf0a.png)

Mean Squared Error: 13.494105815760902

kernel = DotProduct() + WhiteKernel()

Acknowledgement:
World Bank Datasets
