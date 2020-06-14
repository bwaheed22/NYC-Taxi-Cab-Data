#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:14:46 2020

@author: joemarlo
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *

os.chdir('/home/joemarlo/Dropbox/Data/Projects/NYC-Taxi-Cab-Data')
conn = sqlite3.connect('NYC-Taxi.db')

# =============================================================================
# EDA
# =============================================================================

# define cursor
cur = conn.cursor()

# get nrow of data
cur.execute(
    "SELECT COUNT(*) FROM green_cabs;"
    )
cur.fetchall()

cur.execute(
    "SELECT COUNT(*) FROM yellow_cabs;"
    )
cur.fetchall()

# get column names
cur.execute(
    "PRAGMA table_info(green_cabs);"
    )
col_names = cur.fetchall()
col_names = np.array(col_names)[0:21, 1]

# pull in green cabs
cur.execute(
    "SELECT * FROM green_cabs;"
    )
query_results = cur.fetchall()

green_cabs = pd.DataFrame(query_results, columns = col_names)

del col_names
del cur
del query_results

# density plot
ggplot(green_cabs.sample(n=10000)) +\
 aes(x = 'tip_amount') +\
 geom_density()

# tell pandas to print all the columns
pd.set_option('display.max_columns', None)

# quantiles of data
print(green_cabs.quantile(q=np.array(range(0,11))*0.1))


# filter dataframe for these characteristics
# trip_distance > 0
# fare_amount > 0
# total_amount > 0
# congestion_surcharge > 0
# extra > 0
# payment_type == 1 (credit only)

indices = (green_cabs['trip_distance'] > 0) &\
    (green_cabs['fare_amount'] > 0) &\
    (green_cabs['total_amount'] > 0) &\
    (green_cabs['congestion_surcharge'] >= 0) &\
    (green_cabs['extra'] >= 0) &\
    (green_cabs['payment_type'] == 1)

green_cabs_cleaned = green_cabs[indices]
del indices
green_cabs_cleaned.shape

# quantiles of data
print(green_cabs_cleaned.quantile(q=np.array(range(0,11))*0.1))

# histogram of each feature
green_cabs_cleaned.hist()

