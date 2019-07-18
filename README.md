
# Comparative-Study-of-Regression-Models-on-Temperature-Analysis

## XGBoost Regression:

Ensemble method.

![XGB](https://user-images.githubusercontent.com/37043631/61465058-fd36ed00-a994-11e9-9ba7-3bae23e35d6b.png)

Mean Squared Error: 6.827033810901381

### Parameters:-

n_estimators= 500

learning_rate= 0.05

early_stopping_rounds=5 (To stop overfitting)

## Gaussian Process Regression:

![GP](https://user-images.githubusercontent.com/37043631/61465094-0922af00-a995-11e9-9337-6b85be4cdf0a.png)

Mean Squared Error: 13.494105815760902

kernel = DotProduct() + WhiteKernel()
