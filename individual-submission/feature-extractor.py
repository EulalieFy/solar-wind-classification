from scipy import constants
import pandas as pd
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()

        X_df_new['log_Pdyn'] = np.log(X_df_new.Pdyn)
        
        """delete columns with very low variances"""
        X_df_new=X_df_new.drop(['Range F 14'], axis=1)
        X_df_new=X_df_new.drop(['Pdyn'], axis=1)

        """introduce time series features on selected features and time window, see functions below"""
    
        columns = ['B','Beta','Bx','Vth','By','Bz','RmsBob','Vx']
        for i in ['2h','5h','10h','15h','20h'] : 
            for col in columns: 
                X_df_new = compute_rolling_std(X_df_new, col, i)
                X_df_new = compute_rolling_mean(X_df_new, col, i)

        """to quantify the slope of the features"""
        
        for col in columns: 
            X_df_new[col+'_mean_delta_6h']=X_df_new[col+'_2h_mean'] - X_df_new[col+'_2h_mean'].shift(36)

        
        X_df_new = X_df_new.fillna(0)    
        return X_df_new

def compute_rolling_std(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    time_indow : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    data[name] = data[feature].rolling(time_window, center=center).std()
    data[name] = data[name].ffill().bfill()
    data[name] = data[name].astype(data[feature].dtype)
    return data

def compute_rolling_mean(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    time_indow : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'mean'])
    data[name] = data[feature].rolling(time_window, center=center).mean()
    data[name] = data[name].ffill().bfill()
    data[name] = data[name].astype(data[feature].dtype)
    return data
