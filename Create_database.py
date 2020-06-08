#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:15:46 2020

@author: joemarlo
"""

import os
import sqlite3
import pandas as pd
import glob
import gc

# =============================================================================
# connect to database NYC-Taxi.db; if it doesn't exist this will
#   create it in the working directory
# =============================================================================
os.chdir('/home/joemarlo/Dropbox/Data/Projects/NYC-Taxi-Cab-Data')
conn = sqlite3.connect('NYC-Taxi.db')


# =============================================================================
# read in cabs data, write to database; seperate by color
# =============================================================================

# read green cab files in and write to database
csv_files = glob.glob("Data/green*", recursive = False)
for file in csv_files:
    df = pd.read_csv(file)
    # add identifier
    df['Source_file'] = file
    # write to database
    df.to_sql("green_cabs", conn, if_exists = 'append', index = False)
    # purge memory
    gc.collect()
    # delete csv from directory
    #os.remove(file)

# read yellow cab files in and write to database
csv_files = glob.glob("Data/yellow*", recursive = False)
for file in csv_files:
    df = pd.read_csv(file)
    # add identifier
    df['Source_file'] = file
    # write to database
    df.to_sql("yellow_cabs", conn, if_exists = 'append', index = False)
    # purge memory
    gc.collect()
    # delete csv from directory
    #os.remove(file)


# =============================================================================
# example queries
# =============================================================================

# define cursor
cur = conn.cursor()

# run a query
cur.execute(
    "SELECT tip_amount, trip_distance, fare_amount FROM green_cabs"
    )
query_results = cur.fetchall()
pd.DataFrame(query_results, columns = ['tip_amount', 'trip_distance', 'fare_amount'])

# drop a table
#cur.execute("DROP TABLE green_cabs")
#conn.commit

