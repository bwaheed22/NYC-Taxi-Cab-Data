#!/usr/bin/env python
# coding: utf-8

# # Analysis of NYC Taxi Cab Data
# ## Import libraries and connect to database

# In[1]:


import pandas as pd
import numpy as np
import sqlite3
import os
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt
import seaborn as sns

# set working directory up a level from this file
#os.chdir('..')
os.chdir('/Users/mbp/Documents/Side Projects/NYC-Taxi-Cab-Data')

# Connect to database:
con = sqlite3.connect("NYC-Taxi.db")


# ## Extract observations and clean data
#
# The analysis will focus on a sample of a specified number of observations of green and yellow cab data. Data from each of these sources will be stacked together, features created, and then finally a train and test set created.

# In[2]:


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


# In[3]:


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
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], unit = "s")
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], unit = "s")

df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# ### Remove outliers

# In[6]:


df.quantile(q=np.array(range(0,11))*0.1)


# In[7]:


df.shape


# Based on the above quantiles, the following filters should be applied:
# - trip_distance > 0
# - passenget_count > 0
# - fare_amount > 0
# - extra >= 0
# - mta_tax >= 0
# - tip_amount >= 0
# - tolls_amount >= 0
# - improvement_surcharge >= 0
# - total_amount > 0
# - congestion_surcharge >= 0
#
# The upper limits should also be capped at the 99.9th percentile for:
# - trip_distance
# - fare_amount
# - trip_time
#
# Additionally, we only want to look at credit transactions because cash transactions are less likely to have a tip registered:
# - payment_type == 1

# In[8]:


indices = (df['trip_distance'] > 0) &    (df['passenger_count'] > 0) &    (df['fare_amount'] > 0) &    (df['extra'] >= 0) &    (df['mta_tax'] >= 0) &    (df['tip_amount'] >= 0) &    (df['tolls_amount'] >= 0) &    (df['improvement_surcharge'] >= 0) &    (df['total_amount'] > 0) &    (df['congestion_surcharge'] >= 0) &    (df['payment_type'] == 1) &    (df['trip_distance'] <= df['trip_distance'].quantile(.999)) &    (df['fare_amount'] <= df['fare_amount'].quantile(.999))

df_cleaned = df[indices]
del indices

df_cleaned.shape


# ### Check for NAs

# In[9]:


df_cleaned.isna().sum()


# In[10]:


df = df_cleaned
del df_cleaned


# ## Feature engineering

# ### Add in borough information

# In[11]:


# Store Pickup location borough names and merge to main data frame:
df_location = pd.read_csv("Data/taxi_zones.csv")
df_location['DOLocationID'] = df_location['LocationID']
df_location = df_location.rename(columns = {'LocationID':'PULocationID'})

# Add borough and Zone name:
df = df.merge(df_location[['PULocationID', 'Borough', 'Zone']], on = "PULocationID", how = 'left')


# ### Add weather information

# In[12]:


# Store Weather informaiton and merge to main data frame

df_weather = pd.read_csv("Data/CP.weather.df.csv")

# convert Date to datetime
df_weather['Date'] = pd.to_datetime(df_weather['Date'])
df_weather = df_weather.rename(columns={"Date":"date"})

df['date'] = pd.to_datetime(df['pickup_datetime'].dt.date)

# merge with pickup date
df = df.merge(df_weather, how= 'left')


# In[13]:


# Summary stats and structure of data:
df.head()


# ### Add the following variables
#
# - trip time
# - average speed
# - day of week
# - month
# - hour
# - holiday dummy code
# - tip indicator (1 or 0)

# In[14]:


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

# Create tip indicator
df['tip_ind'] = [0 if x ==0 else 1 for x in df['tip_amount']]


# In[15]:


# Create holiday
cal = calendar()
holidays = cal.holidays(start=df['date'].min(), end=df['date'].max(), return_name=True)
holidays = holidays.reset_index(name='holiday').rename(columns={'index':'date'})
holidays['date'] = pd.to_datetime(holidays['date'])
df = pd.merge(df, holidays, on = 'date', how='left')

# add nye
df['holiday_NYE'] = np.where((pd.to_datetime(df['date']).dt.month == 12) &                      (pd.to_datetime(df['date']).dt.day == 31), 1, 0)


# In[16]:


df.head()


# ## Variable Pre-Processing
#
# * limit df to only features and outcome variable
# * one-hot encode all categorical variables
# * remove reference classes when nessesary
# * Pull out final test set
# * Create Train and Validation sets
# * Create Scale function from training data
# * apply scale function to train, validation and test sets

# In[17]:


df.describe
df.isna().sum()


# In[18]:


# retain only neccessary variables

cols = ['tip_amount', 'tip_ind', 'passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax',
       'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',  'cab', 'Borough',
        'trip_time', 'avg_speed','month', 'day', 'hour','holiday','holiday_NYE', 'Zone',
        'Precipitation', 'Snow.depth', 'Snowfall', 'Max.temp', 'Min.temp','Avg.wind.speed', 'Gust.speed']

df = df[cols]

del cols


# In[19]:


df.columns


# ## Exploratory Data Analysis

# We will examine summary statistics of all variables to get a better look at the structure of the data.

# In[20]:


# Summary stats and structure of data:v
df.describe()


# ## Data Analysis and Visualization

# Now that we've cleaned and inspected the dataset, we can analyze it and use visualizations to get a better understanding of the characteristics of trips between yellow and green cabs, and understand tip amounts amongst all trips.

# ### Summary of Holiday Trips, Trips by Hours of Day, Trips by Day

# In[21]:


days_df = pd.DataFrame(df.day.value_counts()).rename_axis('Day').reset_index()


# In[22]:


plt.figure(figsize=(15,5))
plt.xticks(rotation=45)
sns.barplot(x = days_df['Day'], y = days_df['day'], color = "red").set_title("Number of Trips by Day of Week")


# In[23]:


holidays_df = pd.DataFrame(df.holiday.value_counts()).rename_axis('Holiday').reset_index()
holidays_df


# In[24]:


plt.figure(figsize=(15,5))
plt.xticks(rotation=45)
sns.barplot(x = holidays_df['Holiday'], y = holidays_df['holiday'], color = "blue").set_title("Number of Trips by Holiday")


# In[25]:


hours_df = pd.DataFrame(df.hour.value_counts()).rename_axis('Hour').reset_index()
hours_df


# In[26]:


plt.figure(figsize=(15,5))
plt.xticks(rotation=45)
sns.barplot(x = hours_df['Hour'], y = hours_df['hour'], color = "blue").set_title("Number of Trips by Hour of Day")


# In[27]:


hours_sat = pd.DataFrame(df[df['day']=="Saturday"].hour.value_counts()).rename_axis('Hour').reset_index()
plt.figure(figsize=(15,5))
plt.xticks(rotation=45)

sns.barplot(x = hours_sat['Hour'], y = hours_sat['hour'], color = "red").set_title("Number of Trips by hour of day - Saturday")


# ### Total Fare Amount

# In[28]:


plt.figure(figsize=(15,4))
sns.distplot(df[df['cab']=='green']['fare_amount'], hist=False, color = "green")
sns.distplot(df[df['cab']=='yellow']['fare_amount'], hist=False, color = "y").set_title('Total Fare Amount')


# In[29]:


df.groupby('cab')['fare_amount'].mean()


# On average, the total fare amount for green cabs is slightly greater than yellow cabs. We can check if this is due to longer trip times & higher distances traveled in green cabs than yellow cabs, shown below:

# ### Total Trip Time

# In[30]:


plt.figure(figsize=(15,5))
sns.distplot(df[df['cab']=='green']['trip_time'], hist = False, color = "green")
sns.distplot(df[df['cab']=='yellow']['trip_time'], hist = False, color = "y").set_title('Total Trip Time')


# In[31]:


df.groupby('cab')['trip_time'].mean()


# In[32]:


plt.figure(figsize=(15,5))
sns.distplot(df[df['cab']=='green']['trip_distance'], hist = False, color = "green")
sns.distplot(df[df['cab']=='yellow']['trip_distance'], hist = False, color = "y").set_title('Total Trip Distance')


# In[33]:


df.groupby('cab')['trip_distance'].mean()


# On average, the trip times and distances are slightly higher for green cabs which is consistent with the higher fare amounts.

# ### Total Tip Amount

# In[34]:


plt.figure(figsize=(15,5))
sns.distplot(df[df['cab']=='green']['tip_amount'], hist = False, color = "green")
sns.distplot(df[df['cab']=='yellow']['tip_amount'], hist = False, color = "y").set_title('Total Tip Amount')


# In[35]:


df.groupby('cab')['tip_amount'].mean()


# In[36]:


# Total transactions by cab type:
df[df['tip_amount']>=0].groupby('cab')['tip_amount'].count()


# In[37]:


# Total 0-tip transactions by cab type:
df[df['tip_amount']==0].groupby('cab')['tip_amount'].count()


# * For green cabs, approximately 31% of all transactions don't have tips.
# * For yellow cabs, approximately 4% of all transactions dont have tips.

# ### Trip Speed vs. Tip Amount

# In[38]:


# Pearson correlation coefficient of trip speed and tip amount:
df[['avg_speed','tip_amount']].corr()


# There is a weak correlation between trip speed and the tip amount.

# ## Modeling

# Models to consider:
# * Zero Inflated
# * Random Forest
# * XGBoost
# * Neural Net
# * GAM
# * KNN
# * Logistic Regression (to predict action of tipping)

# In[39]:


# one-hot encode categorical variables

# create dummys without na class
df.hour = df.hour.to_string()
df = pd.get_dummies(df, dummy_na= False,                      columns= ['passenger_count','cab', 'day','hour','month','improvement_surcharge',
                              'congestion_surcharge', 'mta_tax', 'extra'])

# create dummies with na class
df = pd.get_dummies(df, dummy_na= True, columns= ['holiday','Zone','Borough'])

# clean up column names
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('.0$', '')
df.columns = df.columns.str.replace('/', '_')
df.columns = df.columns.str.replace('.', '_')


# ### Split into train validation and test sets

# In[40]:


# pull 10% of data to store as test
test_df = df.sample(frac=0.1, random_state=44)

# remove test set from data
df = df.drop(test_df.index)

# create train set with 70% of remaining cases
train_df = df.sample(frac=.7, random_state=44)

# create validation set
validation_df = df.drop(train_df.index)


# ### Scale Continous Values

# In[41]:


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


# In[42]:


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


# In[43]:


validation_df


# ### Write dfs to Data Directory

# In[44]:


#test_df.to_csv('Data/test.csv')
#validation_df.to_csv('Data/validation.csv')
#train_df.to_csv('Data/train.csv')


# ## Logistic Regression - Tip classification

# In[45]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
display(train_df.isna().sum())


# In[46]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# IMPUTE MISSING VALUES FOR WEATHER DATA
#X = train_df.loc[:, ~train_df.columns.isin(['tip_ind','Precipitation','Snow_depth','Snowfall','Max_temp','Min_temp','Avg_wind_speed','Gust_speed'])]
train_df = train_df.dropna()

X = train_df.iloc[:,2:395]
y = train_df.loc[:,'tip_ind'] # Set target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) # Create train/test split

logitmod = LogisticRegression(random_state = 0).fit(X_train,y_train) # Fit model

tip_preds = logitmod.predict(X_test) # predict


# In[47]:


logitmod.score(X_test,y_test)


# In[48]:


from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix = confusion_matrix(y_test, tip_preds)
print(confusion_matrix)
print(classification_report(y_test, tip_preds))


# In[49]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, tip_preds)
fpr, tpr, thresholds = roc_curve(y_test, logitmod.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# Accuracy ROC = 0.75

# LASSO regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

n_folds = 5
alphas = np.logspace(-4,-0.5,30)
tunedparameters = [{"C":alphas}]

lassomod = LogisticRegression(penalty = "l1", solver="liblinear", random_state = 0, max_iter = 10000)

clf = GridSearchCV(lassomod, tunedparameters, cv = n_folds)
clf.fit(X_train, y_train)
clf.predict(X_test)


# NEXT STEP:
# 1) XGBoost / Random Forest
# 2) Neural Networks
# ## Generalized Additive Model (GAM)

# In[50]:


# Features to include in model:
features = ['trip_distance', 'fare_amount',
        'trip_time', 'avg_speed']


# In[51]:


# Set features (X) and target (y) matrices in the training set:
X = train_df[features]
y = train_df.loc[:,'tip_amount']

# Train-test split to fit the model on 80% of the training data, and test on the remaining 20% of the training data:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[52]:


y_train


# In[53]:


# Fit GAM:
from pygam import LinearGAM, s, f, GAM
pd.set_option('display.max_columns', None)
X_train.head()


# In[54]:


gam = LinearGAM(n_splines=4).gridsearch(X_train.values, y_train.values)


# In[55]:


y_preds = gam.predict(X_test)
plt.plot(y_test, y_preds,"*")
plt.xlabel("Predicted Value")
plt.ylabel("Actual value")


# In[56]:


from sklearn.metrics import mean_squared_error
from math import sqrt

sqrt(mean_squared_error(y_test,y_preds))


# ## K-Nearest Neighbors (KNN)

# In[ ]:


from sklearn import neighbors
knn_reg = neighbors.KNeighborsRegressor(n_neighbors=5)

X = train_df.iloc[:,np.r_[0,2:8]]
y = train_df.loc[:,'tip_ind'] # Set target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) # Create train/test split

knn_reg.fit(X_train, y_train)


# ### Predictions on test set

# In[ ]:


y_preds = knn_reg.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

sqrt(mean_squared_error(y_test,y_preds))


# In[ ]:


plt.scatter(y_test, y_preds)


# In[ ]:


errors = y_preds - y_test
sns.distplot(errors, hist = True)


# In[ ]:


p1 = sns.kdeplot(y_test, shade=True, color="r")
p1 = sns.kdeplot(y_preds, shade=True, color="b")
plt.show()


# ## Linear Regression

# In[57]:


from sklearn.linear_model import LinearRegression

# Instantiate linear regression object
linear_reg = LinearRegression()

# Train with the objt
linear_reg.fit(X_train, y_train)

# Make prediction
lin_preds = linear_reg.predict(X_test)
sqrt(mean_squared_error(y_test,lin_preds))


# In[58]:


plt.plot(y_test, lin_preds,"*")
plt.xlabel("Predicted Value")
plt.ylabel("Actual value")
