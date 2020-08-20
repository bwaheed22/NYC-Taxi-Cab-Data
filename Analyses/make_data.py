import pandas as pd
import numpy as np
import sqlite3
import os
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# set working directory up a level from this file
os.chdir('..')

# Connect to database:
con = sqlite3.connect("NYC-Taxi.db")


# need to set seed by randomly sampling indices in python then pass to sql

# specify number of rows with pull variable
pull = "5000000"
# Store sample of green and yellow cab data into dataframes:
df1 = pd.read_sql(f"SELECT * FROM green_cabs ORDER BY random() LIMIT" + " " +  pull, con=con)
df2 = pd.read_sql(f"SELECT * from yellow_cabs ORDER BY random() LIMIT"+ " " + pull, con=con)

# Add labels for green and yellow cabs and rename pickup/dropoff datetime columns:
df1 = df1.rename(columns={"lpep_pickup_datetime": "pickup_datetime", "lpep_dropoff_datetime":"dropoff_datetime"})
df2 = df2.rename(columns={"tpep_pickup_datetime": "pickup_datetime", "tpep_dropoff_datetime":"dropoff_datetime"})
df1['cab'] = "green"
df2['cab'] = "yellow"


# retain only columns that are in both datasets
cols_to_keep = df1.columns.intersection(df2.columns)
df1 = df1[cols_to_keep]
df2 = df2[cols_to_keep]
del cols_to_keep

# Combine into one data frame:
df = pd.concat([df1, df2], sort=False)
del df1
del df2

# Change format to datetime where necessary:
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df.head()


df.shape

df.quantile(q=np.array(range(0,11))*0.1)

df.shape

indices = (df['trip_distance'] > 0) &\
    (df['passenger_count'] > 0) &\
    (df['fare_amount'] > 0) &\
    (df['extra'] >= 0) &\
    (df['mta_tax'] >= 0) &\
    (df['tip_amount'] >= 0) &\
    (df['tolls_amount'] >= 0) &\
    (df['improvement_surcharge'] >= 0) &\
    (df['total_amount'] > 0) &\
    (df['congestion_surcharge'] >= 0) &\
    (df['payment_type'] == 1) &\
    (df['trip_distance'] <= df['trip_distance'].quantile(.999)) &\
    (df['fare_amount'] <= df['fare_amount'].quantile(.999))

df_cleaned = df[indices]
del indices

df_cleaned.shape

df = df_cleaned
del df_cleaned



# Store Pickup location borough names and merge to main data frame:
df_location = pd.read_csv("Data/taxi_zones.csv")
df_location['DOLocationID'] = df_location['LocationID']
df_location = df_location.rename(columns = {'LocationID':'PULocationID'})

# Add borough and Zone name:
df = df.merge(df_location[['PULocationID', 'borough', 'zone']], on = "PULocationID", how = 'left')


# Store Weather informaiton and merge to main data frame

df_weather = pd.read_csv("Data/CP.weather.df.csv")

# convert Date to datetime
df_weather['Date'] = pd.to_datetime(df_weather['Date'])
df_weather = df_weather.rename(columns={"Date":"date"})

df['date'] = pd.to_datetime(df['pickup_datetime'].dt.date)

# merge with pickup date
df = df.merge(df_weather, how= 'left')


# Include trip time:
df['trip_time'] = (df['dropoff_datetime'] - df['pickup_datetime']).astype('timedelta64[m]')
# filter out impossible trip times
df = df[df['trip_time'] > 0]

# Create average speed
df['avg_speed'] = df["trip_distance"]/(df["trip_time"]/60)

# Create day of pickup
df['day'] = df['pickup_datetime'].dt.day_name()

# Create hour of pick up
df['hour'] = df['pickup_datetime'].dt.hour

# Create month of pick up
df['month'] = df['pickup_datetime'].dt.month_name()


# Create holiday
cal = calendar()
holidays = cal.holidays(start=df['date'].min(), end=df['date'].max(), return_name=True)
holidays = holidays.reset_index(name='holiday').rename(columns={'index':'date'})
holidays['date'] = pd.to_datetime(holidays['date'])
df = pd.merge(df, holidays, on = 'date', how='left')

# add nye
df['holiday_NYE'] = np.where((pd.to_datetime(df['date']).dt.month == 12) & \
                     (pd.to_datetime(df['date']).dt.day == 31), 1, 0)


# retain only neccessary variables

cols = ['tip_amount', 'passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax',
      'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',  'cab', 'borough',
       'trip_time', 'avg_speed','month', 'day', 'hour','holiday','holiday_NYE', 'zone',
       'Precipitation', 'Snow.depth', 'Snowfall', 'Max.temp', 'Min.temp','Avg.wind.speed', 'Gust.speed']

df = df.loc[:, cols]

del cols


# one-hot encode categorical variables

# create dummys without na class
df = pd.get_dummies(df, dummy_na= False,  \
                    columns= ['passenger_count','cab', 'day','hour','month','improvement_surcharge',
                              'congestion_surcharge', 'mta_tax', 'extra'])

# create dummies with na class
df = pd.get_dummies(df, dummy_na= True, columns= ['holiday','zone','borough'])

# clean up column names
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('.0$', '')
df.columns = df.columns.str.replace('/', '_')
df.columns = df.columns.str.replace('.', '_')

# pull 10% of data to store as test
test_df = df.sample(frac=0.1, random_state=44)

# remove test set from data
df = df.drop(test_df.index)

# create train set with 70% of remaining cases
train_df = df.sample(frac=.7, random_state=44)

# create validation set
validation_df = df.drop(train_df.index)

# obtain scales from train set

from sklearn import preprocessing

continous = train_df[['trip_distance', 'fare_amount','tolls_amount',
                      'trip_time','avg_speed','Precipitation',
                      'Snow_depth', 'Snowfall','Max_temp',
                      'Min_temp','Avg_wind_speed','Gust_speed']]

scaler = preprocessing.MinMaxScaler().fit(continous)
continous = scaler.transform(continous)


train_df[['trip_distance', 'fare_amount','tolls_amount',
                      'trip_time','avg_speed','Precipitation',
                      'Snow_depth', 'Snowfall','Max_temp',
                      'Min_temp','Avg_wind_speed','Gust_speed']] = continous


# apply scale to validation and test set

# validation
from sklearn import preprocessing

apply_scale = validation_df[['trip_distance', 'fare_amount','tolls_amount',
                      'trip_time','avg_speed','Precipitation',
                      'Snow_depth', 'Snowfall','Max_temp',
                      'Min_temp','Avg_wind_speed','Gust_speed']]

apply_scale = scaler.transform(apply_scale)


validation_df[['trip_distance', 'fare_amount','tolls_amount',
                      'trip_time','avg_speed','Precipitation',
                      'Snow_depth', 'Snowfall','Max_temp',
                      'Min_temp','Avg_wind_speed','Gust_speed']] = apply_scale

# test
apply_scale = test_df[['trip_distance', 'fare_amount','tolls_amount',
                      'trip_time','avg_speed','Precipitation',
                      'Snow_depth', 'Snowfall','Max_temp',
                      'Min_temp','Avg_wind_speed','Gust_speed']]

apply_scale = scaler.transform(apply_scale)


test_df[['trip_distance', 'fare_amount','tolls_amount',
                      'trip_time','avg_speed','Precipitation',
                      'Snow_depth', 'Snowfall','Max_temp',
                      'Min_temp','Avg_wind_speed','Gust_speed']] = apply_scale
