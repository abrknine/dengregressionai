from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import math

from matplotlib import pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import statsmodels.api as sm

from scipy import stats

from sklearn.ensemble import RandomForestRegressor

from warnings import filterwarnings
filterwarnings('ignore')

filepath = 'data'
X = pd.read_csv(filepath + '/dengue_features_train.csv')
Y = pd.read_csv(filepath + '/dengue_labels_train.csv')
T = pd.read_csv(filepath + '/dengue_features_test.csv')

# concating total cases to train data frame - beacuse of issue in removing outliers
X = pd.concat([Y['total_cases'], X], axis=1)
# remove correlated columns
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if np.absolute(corr_matrix.iloc[i, j]) >= threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
                    
NaNDic = (X.isnull().sum()*100/X.shape[0])>=10
for i in X.columns.values:
    if(NaNDic[i]):
        X.drop(i,axis=1,inplace=True)
        T.drop(i,axis=1,inplace=True)
        
        
# seperate into two cities
X_sj = X[X['city'] == "sj"]
X_iq = X[X['city'] == "iq"]
T_sj = T[T['city'] == "sj"]
T_iq = T[T['city'] == "iq"]
# drop columns
dropping_columns = ['city']
X_sj = X_sj.drop(dropping_columns, axis=1)
X_iq = X_iq.drop(dropping_columns, axis=1)
T_sj = T_sj.drop(dropping_columns, axis=1)
T_iq = T_iq.drop(dropping_columns, axis=1)
# fill NaN values
X_sj.interpolate(inplace=True)
X_iq.interpolate(inplace=True)
T_sj.interpolate(inplace=True)
T_iq.interpolate(inplace=True)


# remove outliers
X_sj = X_sj[(np.abs(stats.zscore(X_sj.drop(['year','weekofyear','week_start_date','total_cases'],axis=1))) < 5).all(axis=1)]
X_iq = X_iq[(np.abs(stats.zscore(X_iq.drop(['year','weekofyear','week_start_date','total_cases'],axis=1))) < 5).all(axis=1)]

# sperating total_cases label again
L_sj = pd.DataFrame(X_sj['total_cases'])
L_iq = pd.DataFrame(X_iq['total_cases'])

# drop total_cases and back X_sj,X_iq in dataset
X_sj = X_sj.drop(['total_cases'],axis=1)
X_iq = X_iq.drop(['total_cases'],axis=1)

# concating test and train
XandT_sj = pd.concat([X_sj, T_sj])
XandT_iq = pd.concat([X_iq, T_iq])

# stores droped columns from XandT_sj, XandT_iq
XandT_sj_rest = pd.DataFrame(XandT_sj[['year','week_start_date']])
XandT_iq_rest = pd.DataFrame(XandT_iq[['year','week_start_date']])

# drop to normalize
XandT_sj.drop(['year','week_start_date'], axis=1, inplace=True)
XandT_iq.drop(['year','week_start_date'], axis=1, inplace=True)

# scaling training set with test set together
XandT_sj[XandT_sj.columns] = MinMaxScaler().fit_transform(XandT_sj)
XandT_iq[XandT_iq.columns] = MinMaxScaler().fit_transform(XandT_iq)

XandT_sj = pd.concat([XandT_sj_rest, XandT_sj], axis=1)
XandT_iq = pd.concat([XandT_iq_rest, XandT_iq], axis=1)

# final scaled X
X_sj = XandT_sj[:926]
X_iq = XandT_iq[:515]

# final scaled T
T_sj = XandT_sj[926:]
T_iq = XandT_iq[515:]

# correlations
sj_correlations = pd.concat([X_sj, L_sj], axis=1).corr().total_cases.drop('total_cases')
iq_correlations = pd.concat([X_iq, L_iq], axis=1).corr().total_cases.drop('total_cases')


# low results
for i in X_sj.drop(['year','week_start_date'], axis=1).columns.values:
    X_sj[i] = X_sj[i] * np.absolute(sj_correlations[i]) * 100
    X_iq[i] = X_iq[i] * np.absolute(iq_correlations[i]) * 100
    
forest_model_sj = RandomForestRegressor(n_estimators=10000)
forest_model_sj.fit(X_sj.drop(['week_start_date','year'], axis=1), L_sj)

forest_model_iq = RandomForestRegressor(n_estimators=10000)
forest_model_iq.fit(X_iq.drop(['week_start_date','year'], axis=1), L_iq)

forest_predict_sj = forest_model_sj.predict(T_sj.drop(['week_start_date','year'], axis=1))
forest_predict_iq = forest_model_iq.predict(T_iq.drop(['week_start_date','year'], axis=1))

predict_list = list((forest_predict_sj).astype(int)) + list((forest_predict_iq).astype(int))

S = pd.read_csv(filepath + '/submission_format.csv')

S['total_cases'] = predict_list

S.to_csv('data/forest_normalized_weighted.csv', index=False)