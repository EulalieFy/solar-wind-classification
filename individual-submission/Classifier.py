from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
import xgboost as xgb
import numpy as np


class Classifier(BaseEstimator):
    def __init__(self):

        # hyperparameters have been found with gridsearchs
        xgboost_model = xgb.XGBRegressor(booster = 'gbtree',objective = 'binary:logistic', colsample_bytree = 0.9, learning_rate = 0.1,
                max_depth = 5, alpha =10, n_estimators = 50 , eval_metric = 'auc')
        
        #normalize the data
        self.model = make_pipeline(StandardScaler(), xgboost_model)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        pred = self.model.predict(X)
        y_pred = np.asarray([[1-i, i] for i in pred])

        #post processing of the predictions : smoothing
        y_pred_smooth = pd.DataFrame(data = y_pred).rolling(7, min_periods =0, center=True).quantile(0.90).values
        return y_pred_smooth