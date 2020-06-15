# Import libraries for data analysis and connection to sqlite database:

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Connect to database:
con = sqlite3.connect("NYC-Taxi.db")

# Store sample of green and yellow cab data into dataframes:
rsample = np.random.randint(500000)
df1 = pd.read_sql(f"SELECT * FROM green_cabs ORDER BY random() LIMIT {rsample}", con)
df2 = pd.read_sql(f"SELECT * from yellow_cabs ORDER BY random() LIMIT {1000000-rsample}", con)

    # Clean data:
    df1 = df1.rename(columns={"lpep_pickup_datetime": "pickup_datetime", "lpep_dropoff_datetime":"dropoff_datetime"})
    df2 = df2.rename(columns={"tpep_pickup_datetime": "pickup_datetime", "tpep_dropoff_datetime":"dropoff_datetime"})
    df1['cab'] = "green"
    df2['cab'] = "yellow"

    # Combine into one data frame:
    frames = [df1, df2]
    df = pd.concat(frames)

    # Change format to datetime where necessary:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],unit='s')
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'],unit='s')

# Feature Engineering:

    # Store location ID borough names and merge to main data frame:
    df_location = pd.read_csv("/Users/mbp/Documents/Side Projects/NYC Taxi Cab Scratch/taxi+_zone_lookup.csv")
    df_location['DOLocationID'] = df_location['LocationID']
    df_location = df_location.rename(columns = {'LocationID':'PULocationID'})

    df = df.merge(df_location[['PULocationID', 'Borough', 'Zone']], on = "PULocationID", how = 'left')

    # Include trip time:
    df['trip_time'] = (df['dropoff_datetime'] - df['pickup_datetime']).astype('timedelta64[m]')


# Exploratory data analysis (EDA)

    # Summary stats and structure of data:
    df.isna().sum()
    df.describe()
    df.info()
    df.columns

    # Remove outliers / trim dataset:
    qlow = df.quantile(0.01)
    qhigh = df.quantile(0.99)
    df_out = df.iloc[:,3:25]

    # Cleaned data without outliers:
    df_out = df_out[~((df_out < qlow) |(df_out > qhigh)).any(axis=1)]

    plt.hist(df_out['payment_type'])
    plt.hist(df_out['Borough'])
    plt.show()

    sns.distplot(df_out['total_amount'], hist=False)
    sns.distplot(df_out['tip_amount'], hist=False)
    sns.distplot(df_out['trip_distance'], hist=False)


