library(tidyverse)
library(RSQLite)

# this script takes the downloaded data and builds a single SQLite database from it
# must run download_instruction.sh shell scripts to download the data
#  before running this script

work_dir <- "/home/joemarlo/Dropbox/Data/Projects/NYC-Taxi-Cab-Data/"

# connect to database -----------------------------------------------------

# connect to database NYC-Taxi.db; if it doesn't exist this will
#  create it in the working directory
conn <- dbConnect(RSQLite::SQLite(), paste0(work_dir, "NYC-Taxi.db"))


# read in green cabs, write to database ------------------------------------

csv_files <- list.files(
  path = paste0(work_dir, "Data/"),
  pattern = '^green(.+).csv$')

for (file in csv_files){
  # read file in
  dat <- read_csv(file = paste0(work_dir, "Data/", file),
                  col_types = 'iTTciiiidddddddddiid')
  # add identifier
  dat$Source_file <- file
  # add to database
  dbWriteTable(
    conn = conn,
    name = "green_cabs",
    value = dat,
    append = TRUE
  )
  rm(dat)
  gc()
}
rm(file, csv_files)


# read in yellow cabs, write to database ------------------------------------

csv_files <- list.files(
  path = paste0(work_dir, "Data/"),
  pattern = '^yellow(.+).csv$')

for (file in csv_files){
  # read file in
  dat <- read_csv(file = paste0(work_dir, "Data/", file),
                  col_types = 'iTTidiciiidddddddd')
  # add identifier
  dat$Source_file <- file
  # add to database
  dbWriteTable(
    conn = conn,
    name = "yellow_cabs",
    value = dat,
    append = TRUE
  )
  rm(dat)
  gc()
}
rm(file, csv_files)


# example queries ---------------------------------------------------------

# list all the tables available in the database
dbListTables(conn)

# collect()'ing the table brings it into memory
tbl(conn, "green_cabs") %>% head() %>% collect() %>% View

# test a query
tbl(conn, "green_cabs") %>%
  group_by(Source_file) %>%
  summarize(n.rows = n()) #%>% 
  # collect()

# check number of rows per table for all tables starting with 'green'
dbListTables(conn) %>%
  grep("^green*", ., value = T) %>%
  sapply(., function(table) {
    tbl(conn, table) %>%
      tally() %>%
      collect() %>%
      as.numeric()
  })

# remove all tables from the database
# sapply(dbListTables(conn), function(table) dbRemoveTable(conn, table))

# remove just ^green tables
# dbListTables(conn) %>% grep("^green", ., value = T) %>% sapply(., function(table) dbRemoveTable(conn, table))

# can also run (most) standard SQL queries
dbGetQuery(conn, 'SELECT tip_amount, trip_distance,  "fare_amount"
                  FROM "green_cabs"
                  WHERE trip_distance < 1')
