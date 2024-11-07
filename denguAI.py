
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:19:25 2019

@author: HP
"""

import re
import time
import datetime 
import operator
import numpy as np
import pandas as pd 
import collections
import unicodedata
import collections
import seaborn as sns
import collections
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor

from tqdm import tqdm
from collections import Counter
from datetime import datetime, date, timedelta
from IPython.display import Image

features_train = pd.read_csv("data/dengue_features_train.csv")
features_test = pd.read_csv("data/dengue_features_test.csv")
labels_train = pd.read_csv("data/dengue_labels_train.csv")

#merge the training features with labels
train = pd.merge(labels_train, features_train, on=['city','year','weekofyear'])
#check whether there is any duplicate labels
#print(np.sum(train.duplicated()))


'''sns.set(style="ticks", palette="colorblind")
g = sns.FacetGrid(train, col="city",aspect=2)  
g.map(sns.distplot, "total_cases") 
axes = g.axes
axes[0,0].set_ylim(0,0.090)
axes[0,1].set_ylim(0,0.090)

train.groupby('city').mean().total_cases

sns.set(style="ticks", palette="colorblind")
fig = sns.FacetGrid(train, hue='city', aspect=4) 
fig.map(sns.pointplot,'weekofyear','total_cases')
max_x = train.weekofyear.max()
min_x = train.weekofyear.min()
fig.set(xlim=(min_x,max_x))
fig.set(ylim=(0, 80))
fig.add_legend()'''


train.isnull().sum()

features_test.isnull().sum()

train.drop('year', axis=1, inplace=True)
train.drop('week_start_date', axis=1, inplace=True)

train_sj = train[train.city == 'sj'].copy()
train_iq = train[train.city == 'iq'].copy()

#x_train_sj = features_train[features_train.city == 'sj'].copy()
#x_train_iq = features_train[features_train.city == 'iq'].copy()

test_sj = features_test[features_test.city == 'sj'].copy()
test_iq = features_test[features_test.city == 'iq'].copy()

train_sj.isnull().sum()


features = ['ndvi_ne', 'ndvi_nw','ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm']

def fillMissingValues(df, imputer):
    
    imputer.fit(df[features])
    df[features] = imputer.transform(df[features])
    return df

imputer_sj = Imputer(strategy = 'mean')
train_sj = fillMissingValues(train_sj, imputer_sj)
test_sj = fillMissingValues(test_sj, imputer_sj)

imputer_iq = Imputer(strategy = 'mean')
train_iq = fillMissingValues(train_iq, imputer_iq)
test_iq = fillMissingValues(test_iq, imputer_iq)

def transformTemperatureValues(df):
    
    df['reanalysis_air_temp_k'] = df.reanalysis_air_temp_k -273.15
    df['reanalysis_avg_temp_k'] = df.reanalysis_avg_temp_k-273.15
    df['reanalysis_dew_point_temp_k'] = df.reanalysis_dew_point_temp_k-273.15
    df['reanalysis_max_air_temp_k'] = df.reanalysis_max_air_temp_k-273.15
    df['reanalysis_min_air_temp_k'] = df.reanalysis_min_air_temp_k-273.15
    
    return df

train_sj = transformTemperatureValues(train_sj)
train_iq = transformTemperatureValues(train_iq)
test_sj = transformTemperatureValues(test_sj)
test_iq = transformTemperatureValues(test_iq)

sj_correlations = train_sj.corr()
iq_correlations = train_iq.corr()

import matplotlib.cm as cm
from matplotlib import cm
cmap = cmap=sns.diverging_palette(5, 250, as_cmap = True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

sj_correlations.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())

(sj_correlations
             .total_cases
             .drop('total_cases')
             .sort_values(ascending = False)
             .plot
             .barh())
sns.set(style="ticks", palette= "colorblind")

'''iq_correlations.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())

(iq_correlations
             .total_cases
             .drop('total_cases')
             .sort_values(ascending = False)
             .plot
             .barh())
sns.set(style="ticks", palette= "colorblind")'''

importantFeature = ['reanalysis_specific_humidity_g_per_kg', 
                    'reanalysis_dew_point_temp_k', 
                    'reanalysis_min_air_temp_k',
                    'station_min_temp_c',
                    'station_max_temp_c',
                    'station_avg_temp_c']

dropFeatures = list(set(features) - set(importantFeature))

def droppingFeatures(df):
    df.drop(dropFeatures, axis=1, inplace=True)
    return df

train_sj = droppingFeatures(train_sj)
train_iq = droppingFeatures(train_iq)
test_sj = droppingFeatures(test_sj)
test_iq = droppingFeatures(test_iq)

def normalizeData(feature):
    return (feature - feature.mean()) / feature.std()

train_sj[importantFeature] = train_sj[importantFeature].apply(normalizeData, axis=0)
train_iq[importantFeature] = train_iq[importantFeature].apply(normalizeData, axis=0)
test_sj[importantFeature] = test_sj[importantFeature].apply(normalizeData, axis=0)
test_iq[importantFeature] = test_iq[importantFeature].apply(normalizeData, axis=0)
    

sj_train_subtrain = train_sj.head(800)
sj_train_subtest = train_sj.tail(train_sj.shape[0] - 800)

iq_train_subtrain = train_iq.head(400)
iq_train_subtest = train_iq.tail(train_iq.shape[0] - 400)

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf


import statsmodels.api as sm


from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf

def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "reanalysis_min_air_temp_k + " \
                    "station_min_temp_c + " \
                    "station_max_temp_c + " \
                    "station_avg_temp_c"
                    
    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
    print(grid)
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)
        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model
    
sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)

figs, axes = plt.subplots(nrows=2, ncols=1)

# plot sj
train_sj['fitted'] = sj_best_model.fittedvalues
train_sj.fitted.plot(ax=axes[0], label="Predictions")
train_sj.total_cases.plot(ax=axes[0], label="Actual")

# plot iq
train_iq['fitted'] = iq_best_model.fittedvalues
train_iq.fitted.plot(ax=axes[1], label="Predictions")
train_iq.total_cases.plot(ax=axes[1], label="Actual")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()


sj_predictions = sj_best_model.predict(test_sj).astype(int)
iq_predictions = iq_best_model.predict(test_iq).astype(int)

'''submission = pd.read_csv("data/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("data/Model_2.csv")'''



