
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar



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
df = pd.concat(frames, sort=False)

# Change format to datetime where necessary:
df.loc[:, ['pickup_datetime','dropoff_datetime']] = df.loc[:, ['pickup_datetime','dropoff_datetime']].apply(pd.to_datetime, errors='coerce')

## Feature Engineering:

# Include trip time:
df['trip_time'] = (df['dropoff_datetime'] - df['pickup_datetime']).astype('timedelta64[m]')

# Create average speed
df['avg_speed'] = df["trip_distance"]/(df["trip_time"]/60)

# Create day of pickup
df['day'] = df['pickup_datetime'].dt.day_name()

# did pickup occur on a holiday 
cal = calendar()
df['date'] = pd.to_datetime(df['dropoff_datetime']).dt.date
holidays = cal.holidays(start=df['date'].min(), end=df['date'].max(), return_name=True)
holidays = holidays.reset_index(name='holiday').rename(columns={'index':'date'})
holidays['date'] = pd.to_datetime(holidays['date']).dt.date
df = pd.merge(df, holidays, on = 'date', how='left')
# create dummys for holiday 
df = pd.get_dummies(df,dummy_na= True, columns= ['holiday'])

# add nye
df['holiday_NYE'] = np.where((pd.to_datetime(df['date']).dt.month == 12) & \
                     (pd.to_datetime(df['date']).dt.day == 31), 1, 0)

# hour of day variable
hour = pd.get_dummies(df['pickup_datetime'].dt.hour, dummy_na= True)
hour.columns = ['hour_' + str(col) for col in hour.columns]
df = pd.concat([df, hour], axis=1)
