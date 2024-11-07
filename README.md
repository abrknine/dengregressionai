# DengAI

## Introduction
DengAI is a competition hosted by DrivenData Foundation to solve a critical problem which caused by mosquitoes with the use of historical data containing details about the environmental conditions to predict dengue disease spread. Dengue fever is a mosquito-borne deadly viral disease that occurs in tropical and subtropical parts of the world. Since this viral disease is carried by mosquitoes, the spreading of dengue is related to the climatic variables like temperature, precipitation, humidity etc. Therefore with the correlation between these environmental conditions and historical data it is possible to predict the dengue cases.
This report contains details about the methods and techniques we have used to predict the dengue disease spread.

## Problem Statement
Our task in this project is to predict the total number of dengue infected patients in a given week of the year for the given location (two cities: San Juan and Iquitos) based on the provided climate data such as daily climate data weather station measurements, satellite precipitation measurements, NOAA's NCEP Climate Forecast System Reanalysis measurements and Satellite vegetation index data. 
## Objectives
The competition, “DengAI: Predicting Disease Spread” is hosted by DrivenData and it is about predicting the total dengue cases each week in two loactions (San Juan and Iquitos) based on the given environmental conditions.

The objective of this project is to research and build an appropriate model with a low Mean Absolute Error (MAE) for predicting the total dengue cases in the test set provided by the DrivenData, “DengAI: Predicting Disease Spread” competition.
## Motivation
Dengue fever is a mosquito-borne deadly viral disease that occurs in tropical and subtropical parts of the world.
This viral disease is carried by mosquitoes, the spreading of dengue is related to the climatic variables like temperature, precipitation, humidity and etc.
Therefore with the correlation between these environmental conditions and historical data, it is possible to predict the dengue cases.
## Methodology
The following are the major steps that we followed to build a model with better accuracy in predicting the number of total cases.
1. Data Analysis
2. Data Preprocessing
3. Build a model
4. Training and evaluation
5. Predicting test data

## Dataset
There are mainly three data files provided by the DrivenData competition.

    1. Training Data Features
Contains the features and the relevant attribute values of the training dataset
Consists of 24 features and attribute values for 1456 weeks in both San Juan and Iquitos cities

    2. Training Data Labels
Contains the total number of dengue cases for each row in the training dataset

    3. Test Data Features
Contains the features and the relevant attribute values of testing data set
Contains the same 24 features as the training feature set
There are 416 weeks for San Juan and Iquitos cities need to predict the total_cases

Both of the provided training and testing feature datasets consist with climate data as well as non-climate data on a year, weekOfYear timescale. The data is available in following categories.

### City and Date indicators:
This data describes the considered cities and the starting date of the data.
* city - City abbreviations: ‘iq’ for Iquitos and ‘sj’ for San Juan
* week_start_date - The week start date given in yyyy-mm-dd format

### NOAA's GHCN daily climate data weather station measurements
The Global Historical Climatology Network (GHCN) is an integrated database which consists of daily climate reports from land surface stations all over the world. This data includes the records over 100,000 stations in 180 countries and territories across the world. The data provided by GHCN is subjected to a common suite of quality assurance reviews. Following attributes are available in the dataset from category of data.
* station_max_temp_c – Maximum temperature
* station_min_temp_c – Minimum temperature
* station_avg_temp_c – Average temperature
* station_precip_mm – Total precipitation
* station_diur_temp_rng_c – Diurnal temperature range

### PERSIANN satellite precipitation measurements (0.25x0.25-degree scale)
The Precipitation Estimation from Remotely Sensed Information using Artificial Neural Networks- Climate Data Record (PERSIANN-CDR) is the rainfall estimation resolution of the given degree scale (0.25x0.25- degree scale). Following attribute is available in the dataset from the titled category of data.
* precipitation_amt_mm – Total precipitation

### NOAA's NCEP Climate Forecast System Reanalysis measurements (0.5x0.5-degree scale)
The reanalysis data provided by the National Centers for Environmental Prediction (NCEP) represents how weather and climate change with time. This data is useful for comparing present and past climate conditions. The following attributes from this category have been included in the provided datasets by the competition.
* reanalysis_sat_precip_amt_mm – Total precipitation
* reanalysis_dew_point_temp_k – Mean dew point temperature
* reanalysis_air_temp_k – Mean air temperature
* reanalysis_relative_humidity_percent – Mean relative humidity
* reanalysis_specific_humidity_g_per_kg – Mean specific humidity
* reanalysis_precip_amt_kg_per_m2 – Total precipitation
* reanalysis_max_air_temp_k – Maximum air temperature
* reanalysis_min_air_temp_k – Minimum air temperature

### Satellite vegetation - Normalized difference vegetation index (NDVI) - NOAA's CDR Normalized Difference Vegetation Index (0.5x0.5-degree scale) measurements
The provided data under this category represents a measure of vegetation in a considered area. These values are useful to determine the availability of vegetation and water in an area over time. Following features of the dataset are fallen under this category.

* ndvi_se – Pixel southeast of city centroid
* ndvi_sw – Pixel southwest of city centroid
* ndvi_ne – Pixel northeast of city centroid
* ndvi_nw – Pixel northwest of city centroid

From the correlation analysis, we have dropped the following features which are least correlated attributes.


* ndvi_ne
* reanalysis_relative_humidity_percent
* station_precip_mm
* ndvi_sw
* reanalysis_tdtr_k
* ndvi_se
* station_diur_temp_rng_c
* precipitation_amt_mm
* reanalysis_avg_temp_k
* reanalysis_max_air_temp_k
* reanalysis_air_temp_k
* reanalysis_min_air_temp_k
* reanalysis_sat_precip_amt_mm
* ndvi_nw

## Data Preprocessing

### Data Cleaning
* Imputation of Missing Values
When analyzing the provided data we observed that most of the features have some missing values and we identified several techniques to impute the missing values such as filling them with forward or backward fill (ffill and bfill), interpolation and filling missing values with mean, median or mode and etc. 
Here we used redundant temperature features to impute missing temperature values in the dataset. First we converted reanalysis and station temperature features to Celsius scale and used them to fill the missing temperature values as following example.
Ex: 
df['reanalysis_avg_temp_c'] = df.reanalysis_avg_temp_k - 273.15
df.reanalysis_avg_temp_c -= (df.reanalysis_avg_temp_c - df.station_avg_temp_c).mean()
df.loc[df.station_avg_temp_c.isnull(), 'station_avg_temp_c'] = df.reanalysis_avg_temp_c
Using above technique we filled the missing values of features: ‘station_avg_temp_c’, ‘station_max_temp_c’ and ‘station_min_temp_c’.
Missing values in ‘precipitation_amt_mm_last_week’ feature are filled with backward filling. Missing values in other features were imputed with feature mean. 

### Identify and remove outliers
Calculated the Z-score values and checked whether there are fall in between given threshold value and if not they were dropped from the dataset.
### Data Transformation
In the dataset, the station temperatures were given in Celsius scale while reanalysis temperature features were given in Kelvin scale. As a data transformation step, all of those were converted into the Celsius scale to avoid confusion. Then performed Z-score normalization using feature mean and feature stand deviation.
### Data Reduction
Although the original datasets were provided with 24 features, we only used a selected set of features for our models. We analyzed the correlations of features with the tola_cases and obtained the following results.

According to the correlation analysis we were able to identify the highly correlated features and selected them to use in our model as the input features. So, we identified and selected the following features and other features were dropped from the datasets. The selected set of features are:
reanalysis_dew_point_temp_k
station_min_temp_c
station_max_temp_c
Station_avg_temp_c
reanalysis_precip_amt_kg_per_m2
reanalysis_specific_humidity_g_per_kg

### Engineer new features
We have engineered several new features other than the provided features such as: 
precipitation_amt_last_month
recent_mean_dew_point
recent_mean_spec_humidity
recent_precipitation_sum

The total precipitation of last month has a higher impact to dengue cases than the total precipitation of the considered week. So we engineered the feature ‘precipitation_amt_last_month’ and it was calculated by shifting the ‘precipitation_amt_mm’ feature by 4.
The cumulative effect of prior weeks have a considerable impact on the number of dengue fever cases in the considered week. So we have engineered ‘recent_mean_dew_point’, ‘recent_mean_spec_humidity’ and ‘recent_precipitation_sum’ by using time series on dew point, humidity and precipitation.
Model
We have built several models to predict the total_cases. This section contains details on the models that we used. From the data analysis we have found that there is a significant difference in data in both cities San Juan and Iquitos. Therefore we use two models to train the data of the two cities. Then tried with various regression algorithms to find the most promising algorithm.
## Negative Binomial
In the challenge, we were supposed to predict the total_cases which is a non negative integer. From the standard regression techniques we select negative binomial as the variance is much higher than the mean in the population distribution of total_cases as shown below. 

San Juan - total_cases

Mean : 34.18055555555556
Variance : 2640.045439691045

Iquitos - total_cases

Mean : 7.565384615384615
Variance : 115.89552393656412

## Random Forest Regressor
Random forest regressor is a flexible and simple machine learning algorithm that can be used for both classification and regression tasks. We have used this model to check whether we could get a better prediction than the other models. The required hyper-parameters for random forest regressor were determined by using grid search. 
## Gradient Boosting Regressor
## Ensemble Model

### Parameter Tuning
The optimal hyper-parameters for best performing models were determined using a grid search. We inputted the ranges for model hyper-parameters and the grid search function returns the optimal parameters among them. The hyper-parameters which were found using the grid search for each model is mentioned below. 

## Random Forest Regressor
#### sj:

n_estimators	: 100

max_depth	: 3

#### iq:

n_estimators	: 300

max_depth	: 3

## Gradient Boosting Regressor

#### sj:

learning_rate	: 0.1

n_estimators	: 10

max_depth	: 3

#### iq:

learning_rate	: 0.1

n_estimators	: 10

max_depth	: 3








