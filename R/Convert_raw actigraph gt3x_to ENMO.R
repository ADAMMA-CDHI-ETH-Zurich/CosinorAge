# MACRO FOR MULTIPLE SEQN - READ IN RAW ACCELEROMTER AND CONVERT TO ENMO
# Load library
# link: https://github.com/muschellij2/SummarizedActigraphy
library(MIMSunit)
# install.packages("~/Downloads/activityCounts_0.1.2.tar.gz", repos = NULL, type = "source")
library(activityCounts) # installed manullay
# remotes::install_github("muschellij2/SummarizedActigraphy")
library(SummarizedActigraphy)
library(data.table)
library(tidyverse)
library(lubridate)

# 1. Read in nhanes_4069_seqn
nhanes_4069_seqn <- readRDS("/Users/jishim/Desktop/NHANES/2023-06/nhanes_4069_seqn.RDS") 
dim(nhanes_4069_seqn)
#3832 1
head(nhanes_4069_seqn)

# set seqn_list
seqn_list <- nhanes_4069_seqn

# keep 2 for now
# seqn_list <- seqn_list[1:6,] # completed
# seqn_list <- seqn_list[7:20, ] # completed
# seqn_list <- seqn_list[21:50, ] # completed
#seqn_list <- seqn_list[51:100, ] # completed
# seqn_list <- seqn_list[101:150, ] # completed
# seqn_list <- seqn_list[150:200, ] # completed
#seqn_list <- seqn_list[3451:3482, ] # completed
#seqn_list <- seqn_list[3483:3615, ] # completed
#seqn_list <- seqn_list[3616:3678, ] # completed
#seqn_list <- seqn_list[3679:3681, ] # completed
#seqn_list <- seqn_list[3682:3797, ] # completed
#seqn_list <- seqn_list[3798:4000, ] # completed
#seqn_list <- seqn_list[3000:3319, ] # completed
#seqn_list <- seqn_list[3321:3441, ]# completed
#seqn_list <- seqn_list[201:400, ]# completed
#seqn_list <- seqn_list[3442:3500, ] # completed
seqn_list <- seqn_list[801:1000, ] # NEED TO RUN!!!

out_all <- data.frame()
for (i in 1:nrow(seqn_list)){ #  seqn_i <- seqn_list[1]
  start.time <- Sys.time()
  # 1. Append csv files into one list
  # GOOD SOURCE: https://martakarass.github.io/post/2021-06-29-pa_measures_and_summarizedactigraphy/
  # Get list of file names
  seqn = seqn_list[i, ]
  print(seqn)
  #file_names <- list.files(path = paste0("/Users/jishim/Desktop/NHANES_raw_30003100/", seqn), pattern = "*.sensor.csv", full.names = TRUE)
  file_names <- list.files(path = paste0("/Users/jishim/Desktop/NHANES_raw/", seqn), pattern = "*.sensor.csv", full.names = TRUE)
  print(head(file_names))
  
  # Read and concatenate all CSV files into a single data.table
  cat('\n', '------ Start concatenating CSV of seqn ------', as.character(seqn),'\n')
  data_all <- rbindlist(lapply(file_names, fread), fill=TRUE) %>% dplyr::select(HEADER_TIMESTAMP, X, Y, Z) %>% mutate(seqn = as.numeric(seqn))
  print(head(data_all))
  cat('------ Completed ------', as.character(seqn),'\n')
  
  # 2. Make POSIXct column named HEADER_TIME_STAMP (80 Hz)
  start_timestamp = substr(data_all$HEADER_TIMESTAMP[1], 1, 19)
  data_all2 <- data_all %>% 
    mutate(HEADER_TIME_STAMP = seq(from = ymd_hms(paste0(start_timestamp)), by = 1/80, length.out = nrow(data_all))) %>%
    dplyr::select(seqn, HEADER_TIME_STAMP, X, Y, Z)
  cat('\n', '------ First 5 obs with POSIXct HEADER_TIME_STAMP ------', as.character(seqn),'\n')
  print(head(data_all2))
  print(dim(data_all2))
  print(get_sample_rate(data_all2))
  # 80 Hz - good
  
  # 3. Keep only if 24 files per day (this will exclude first and last days b/c of incomplete data)
  first_date = date(data_all2$HEADER_TIME_STAMP[1])
  last_date = date(tail(data_all2$HEADER_TIME_STAMP, n=1))

  in_df <- data_all2 %>% 
    mutate(acc_date = date(HEADER_TIME_STAMP)) %>% 
    filter(acc_date != first_date & acc_date != last_date)
  #dim(data_all2)
  cat('\n', '------ First 5 obs with input data frame (excluding first and last days) ------', as.character(seqn),'\n')
  print(head(in_df))
  print(dim(in_df)) # our input data frame
  print(in_df %>% distinct(acc_date))
  print(nrow(in_df %>% distinct(acc_date)))
  
  # 4. Calculate minute-level AC, MIMS, ENMO (by default, unit = "1 min")
  out_df = SummarizedActigraphy::calculate_measures(
    df = in_df,
    sample_rate = 80,
    dynamic_range = c(-6, 6),  # dynamic range of  wGT3x: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/http://s3.amazonaws.com/actigraphcorp.com/wp-content/uploads/2018/02/22094126/GT3X-wGT3X-Device-Manual-110315.pdf
    fix_zeros = FALSE,         # fixes zeros from idle sleep mode -- set FALSE
    calculate_mims = FALSE,     # uses algorithm from MIMSunit package - set TRUE (we can compare with NHANES)
    calculate_ac = FALSE,       # uses algorithm from activityCounts package - set FALSE to reduce processing time
    flag_data = FALSE,         # runs raw data quality control flags algorithm -- set FALSE
    verbose = TRUE)
  out_df <- out_df %>% mutate(acc_date = date(time), seqn = as.numeric(seqn)) %>% 
    dplyr::select(seqn, time, acc_date, ENMO_t)
  cat('\n', '------ First 5 obs with minute-level ENMO output data frame  ------', as.character(seqn),'\n')
  print(dim(out_df))
  print(head(out_df))
  
  # 5. Save data
  saveRDS(out_df, paste0("/Users/jishim/Desktop/NHANES_enmo/", as.character(seqn), ".RDS"))
  # write.csv(out_df, paste0("/Users/jishim/Desktop/NHANES_enmo/", as.character(seqn), ".csv"))
  cat('\n', '------ SUCCESSFUL!  ------', as.character(seqn),'\n')
  end.time <- Sys.time()
  print(round(end.time-start.time, 2))
}




### END OF PROGRAM ###


### working with single file 
# setwd("/Users/jishim/Desktop/NHANES_raw_30003100/62161")
a1 <- read.csv("/Users/jishim/Desktop/NHANES_raw_30003100/62209/GT3XPLUS-AccelerationCalibrated-1x5x0.NEO1G26599333.2000-01-07-12-30-00-000-P0000.sensor.csv")
head(a1)
# 
# # by = 1/80 hz = 0.0125
# start_timestamp = substr(a1$HEADER_TIMESTAMP[1], 1, 19)
# a2 <- a1 %>% 
#   mutate(HEADER_TIME_STAMP = seq(from = ymd_hms(paste0(start_timestamp)), by = 1/80, length.out = nrow(a1))) %>%
#   dplyr::select(HEADER_TIME_STAMP, X, Y, Z)
# head(a2)
# get_sample_rate(a2)
# # 80 Hz - good
# 
# a2_out = SummarizedActigraphy::calculate_measures(
#   df = a2,
#   sample_rate = 80,
#   dynamic_range = c(-6, 6),  # dynamic range of  wGT3x: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/http://s3.amazonaws.com/actigraphcorp.com/wp-content/uploads/2018/02/22094126/GT3X-wGT3X-Device-Manual-110315.pdf
#   fix_zeros = FALSE,         # fixes zeros from idle sleep mode -- not needed in our case
#   calculate_mims = TRUE,     # uses algorithm from MIMSunit package
#   calculate_ac = TRUE,       # uses algorithm from activityCounts package
#   flag_data = FALSE,         # runs raw data quality control flags algorithm -- not used in our case
#   verbose = FALSE)
# 
# a2_out %>% dplyr::select(HEADER_TIME_STAMP = time, AC, MIMS = MIMS_UNIT, ENMO = ENMO_t, MAD, AI)

# a2_outv2 <- summarize_daily_actigraphy(a2, calculate_mims = TRUE)
# head(a2_outv2) # summarise_daily_actigraphy is the same as calculate_measures just adding a few more quality check columns.


