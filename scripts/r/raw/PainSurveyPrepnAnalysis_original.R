# PainSurveyPrepnAnalysis_original.R
# Purpose: Original analysis script for chronic pain intervention chapter
# Inputs: Merged_Survey_Data.csv (not included in repo)
# Outputs: CSV tables (LMM fixed effects / contrasts), diagnostic plots, exploratory correlation outputs
# Note: Script contains exploratory and duplicate sections from thesis drafting.
# A cleaned, reproducible pipeline will be added in r/intervention/.

# Load the necessary packages
library(tidyverse)
library(lme4)
library(emmeans)
library(multcomp)
library(readr)

# Load the merged dataset
data <- read_csv("Merged_Survey_Data.csv")

# Check structure
glimpse(data)

# View the first few rows
head(data)

table(data$Group)

#remove spaces to clean up the column names and make them consistent 
colnames(data) <- str_replace_all(colnames(data), " ", "")

# Calculate PEG totals for each timepoint
data <- data %>%
  mutate(
    PEG_T1 = rowMeans(data[, c("AvePain1_1", "PEGEnjoyment1_1", "PEGactivity1_1")], na.rm = TRUE),
    PEG_T2 = rowMeans(data[, c("PEGAvePain2_1", "PEGEnjoyment2_1", "PEGactivity2_1")], na.rm = TRUE),
    PEG_T3 = rowMeans(data[, c("PEGAvePain3_1", "PEGEnjoyment3_1", "PEGactivity3_1")], na.rm = TRUE)
  )
colnames(data)

# Recode ISI severity levels
recode_isi <- function(x) {
  recode(x,
         "None" = 0,
         "Mild" = 1,
         "Moderate" = 2,
         "Severe" = 3,
         "Very severe" = 4,
         .default = NA_real_)
}

#apply function to each ISI item for T1
data <- data %>%
  mutate(
    ISI1_1 = recode_isi(ISIseverity1_1),
    ISI1_2 = recode_isi(ISIseverity1_2),
    ISI1_3 = recode_isi(ISIseverity1_3)
  )

# Recode ISI items for T2
data <- data %>%
  mutate(
    ISI2_1 = recode_isi(ISIseverity2_1),
    ISI2_2 = recode_isi(ISIseverity2_2),
    ISI2_3 = recode_isi(ISIseverity2_3)
  )

# Recode ISI items for T3
data <- data %>%
  mutate(
    ISI3_1 = recode_isi(ISIseverity3_1),
    ISI3_2 = recode_isi(ISIseverity3_2),
    ISI3_3 = recode_isi(ISIseverity3_3)
  )

# Calculate total ISI scores (sum of 7 items per ISI scoring instructions)
data <- data %>%
  mutate(
    ISI_Total_T1 = ISI1_1 + ISI1_2 + ISI1_3,
    ISI_Total_T2 = ISI2_1 + ISI2_2 + ISI2_3,
    ISI_Total_T3 = ISI3_1 + ISI3_2 + ISI3_3
  )
#T1 other items
unique(data$ISIsatisfaction1)
unique(data$ISInoticeable1)
unique(data$ISIworried1)
unique(data$ISIdailyFunction1)
#T2 other items 
unique(data$ISIsatisfaction2)
unique(data$ISInoticeable2)
unique(data$ISIworried2)
unique(data$ISIdailyFunction2)
#T3 other items
unique(data$ISIsatisfaction3)
unique(data$ISInoticeable3)
unique(data$ISIworried3)
unique(data$ISIdailyFunction3)

#recode to numeric for all other items
recode_satisfaction <- function(x) {
  recode(x,
         "Very satisfied" = 0,
         "Satisfied" = 1,
         "Moderately satisfied" = 2,
         "Dissatisfied" = 3,
         "Very dissatisfied" = 4,
         .default = NA_real_)
}

recode_noticeable <- function(x) {
  recode(x,
         "Not at all noticeable" = 0,
         "A little noticeable" = 1,
         "Somewhat noticeable" = 2,
         "Quite noticeable" = 3,
         "Very noticeable" = 4,
         .default = NA_real_)
}

recode_worry <- function(x) {
  recode(x,
         "Not at all" = 0,
         "A little" = 1,
         "Somewhat" = 2,
         "Quite" = 3,
         "Very" = 4,
         .default = NA_real_)
}

recode_function <- function(x) {
  recode(x,
         "Not at all" = 0,
         "A little" = 1,
         "Somewhat" = 2,
         "Quite a lot" = 3,
         "Very much" = 4,
         .default = NA_real_)
}

# Recode remaining ISI items for T1
data <- data %>%
  mutate(
    ISI1_4 = recode_satisfaction(ISIsatisfaction1),
    ISI1_5 = recode_noticeable(ISInoticeable1),
    ISI1_6 = recode_worry(ISIworried1),
    ISI1_7 = recode_function(ISIdailyFunction1)
  )

# Recode remaining ISI items for T2
data <- data %>%
  mutate(
    ISI2_4 = recode_satisfaction(ISIsatisfaction2),
    ISI2_5 = recode_noticeable(ISInoticeable2),
    ISI2_6 = recode_worry(ISIworried2),
    ISI2_7 = recode_function(ISIdailyFunction2)
  )

# Recode remaining ISI items for T3
data <- data %>%
  mutate(
    ISI3_4 = recode_satisfaction(ISIsatisfaction3),
    ISI3_5 = recode_noticeable(ISInoticeable3),
    ISI3_6 = recode_worry(ISIworried3),
    ISI3_7 = recode_function(ISIdailyFunction3)
  )

# Calculate ISI total scores for each timepoint (sum of 7 items)
data <- data %>%
  mutate(
    ISI_Total_T1 = ISI1_1 + ISI1_2 + ISI1_3 + ISI1_4 + ISI1_5 + ISI1_6 + ISI1_7,
    ISI_Total_T2 = ISI2_1 + ISI2_2 + ISI2_3 + ISI2_4 + ISI2_5 + ISI2_6 + ISI2_7,
    ISI_Total_T3 = ISI3_1 + ISI3_2 + ISI3_3 + ISI3_4 + ISI3_5 + ISI3_6 + ISI3_7
  )

# Calculate FSS (Fatigue Severity Scale) total scores for each timepoint
data <- data %>%
  mutate(
    FSS_T1 = rowMeans(data[, c('FSQ1_1', 'FSQ1_2', 'FSQ1_3', 'FSQ1_4', 'FSQ1_5', 'FSQ1_6', 'FSQ1_7', 'FSQ1_8', 'FSQ1_9')], na.rm = TRUE),
    FSS_T2 = rowMeans(data[, c('FSQ2_1', 'FSQ2_2', 'FSQ2_3', 'FSQ2_4', 'FSQ2_5', 'FSQ2_6', 'FSQ2_7', 'FSQ2_8', 'FSQ2_9')], na.rm = TRUE),
    FSS_T3 = rowMeans(data[, c('FSQ3_1', 'FSQ3_2', 'FSQ3_3', 'FSQ3_4', 'FSQ3_5', 'FSQ3_6', 'FSQ3_7', 'FSQ3_8', 'FSQ3_9')], na.rm = TRUE)
  )

#HADS scoring checks
# Anxiety items T1
unique(data$HAtense1)
unique(data$HAfrightened1)
unique(data$HAworry1)
unique(data$HAfeelRelaxed1)
unique(data$HAbutterflies1)
unique(data$HArestless1)
unique(data$HApanic1)

# Depression items T1
unique(data$HDenjoy1)
unique(data$HDlaugh1)
unique(data$HDcheerful1)
unique(data$HDslow1)
unique(data$HDappearance1)
unique(data$HDlookFwd1)
unique(data$HDbookTV1)

# Recode HADS Anxiety items at T1
data <- data %>%
  mutate(
    # HAtense1
    HAtense1_score = case_when(
      HAtense1 == "Not at all" ~ 0,
      HAtense1 == "From time to time, occasionally" ~ 1,
      HAtense1 == "A lot of the time" ~ 2,
      HAtense1 == "Most of the time" ~ 3,
      TRUE ~ NA_real_
    ),
    
    # HAfrightened1
    HAfrightened1_score = case_when(
      HAfrightened1 == "Not at all" ~ 0,
      HAfrightened1 == "A little, but it doesn't worry me" ~ 1,
      HAfrightened1 == "Yes, but not too badly" ~ 2,
      HAfrightened1 == "Very definitely and quite badly" ~ 3,
      TRUE ~ NA_real_
    ),
    
    # HAworry1
    HAworry1_score = case_when(
      HAworry1 == "Only occasionally" ~ 0,
      HAworry1 == "From time to time, but not too often" ~ 1,
      HAworry1 == "A lot of the time" ~ 2,
      HAworry1 == "A great deal of the time" ~ 3,
      TRUE ~ NA_real_
    ),
    
    # HAfeelRelaxed1 (reverse scored)
    HAfeelRelaxed1_score = case_when(
      HAfeelRelaxed1 == "Definitely" ~ 3,
      HAfeelRelaxed1 == "Usually" ~ 2,
      HAfeelRelaxed1 == "Not often" ~ 1,
      HAfeelRelaxed1 == "Not at all" ~ 0,
      TRUE ~ NA_real_
    ),
    
    # HAbutterflies1
    HAbutterflies1_score = case_when(
      HAbutterflies1 == "Not at all" ~ 0,
      HAbutterflies1 == "Occasionally" ~ 1,
      HAbutterflies1 == "Quite often" ~ 2,
      HAbutterflies1 == "Very often" ~ 3,
      TRUE ~ NA_real_
    ),
    
    # HArestless1
    HArestless1_score = case_when(
      HArestless1 == "Not at all" ~ 0,
      HArestless1 == "Not very much" ~ 1,
      HArestless1 == "Quite a lot" ~ 2,
      HArestless1 == "Very much indeed" ~ 3,
      TRUE ~ NA_real_
    ),
    
    # HApanic1
    HApanic1_score = case_when(
      HApanic1 == "Not at all" ~ 0,
      HApanic1 == "Not very often" ~ 1,
      HApanic1 == "Quite often" ~ 2,
      HApanic1 == "Very often indeed" ~ 3,
      TRUE ~ NA_real_
    )
  )

# Calculation of HADS Anxiety total score for T1
data <- data %>%
  mutate(
    HADS_Anxiety_T1 = rowSums(cbind(HAtense1_score, HAfrightened1_score, HAworry1_score,
                                    HAfeelRelaxed1_score, HAbutterflies1_score, HArestless1_score, HApanic1_score),
                              na.rm = TRUE)
  )

# Recode HADS Anxiety items for T2
data <- data %>%
  mutate(
    HAtense2_score = case_when(
      HAtense2 == "Not at all" ~ 0,
      HAtense2 == "From time to time, occasionally" ~ 1,
      HAtense2 == "A lot of the time" ~ 2,
      HAtense2 == "Most of the time" ~ 3,
      is.na(HAtense2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HAfrightened2_score = case_when(
      HAfrightened2 == "Not at all" ~ 0,
      HAfrightened2 == "A little, but it doesn't worry me" ~ 1,
      HAfrightened2 == "Yes, but not too badly" ~ 2,
      HAfrightened2 == "Very definitely and quite badly" ~ 3,
      is.na(HAfrightened2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HAworry2_score = case_when(
      HAworry2 == "Only occasionally" ~ 0,
      HAworry2 == "From time to time, but not too often" ~ 1,
      HAworry2 == "A lot of the time" ~ 2,
      HAworry2 == "A great deal of the time" ~ 3,
      is.na(HAworry2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HAfeelRelaxed2_score = case_when(
      HAfeelRelaxed2 == "Definitely" ~ 0,
      HAfeelRelaxed2 == "Usually" ~ 1,
      HAfeelRelaxed2 == "Not often" ~ 2,
      HAfeelRelaxed2 == "Not at all" ~ 3,
      is.na(HAfeelRelaxed2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HAbutterflies2_score = case_when(
      HAbutterflies2 == "Not at all" ~ 0,
      HAbutterflies2 == "Occasionally" ~ 1,
      HAbutterflies2 == "Quite often" ~ 2,
      HAbutterflies2 == "Very often" ~ 3,
      is.na(HAbutterflies2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HArestless2_score = case_when(
      HArestless2 == "Not at all" ~ 0,
      HArestless2 == "Not very much" ~ 1,
      HArestless2 == "Quite a lot" ~ 2,
      HArestless2 == "Very much indeed" ~ 3,
      is.na(HArestless2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HApanic2_score = case_when(
      HApanic2 == "Not at all" ~ 0,
      HApanic2 == "Not very often" ~ 1,
      HApanic2 == "Quite often" ~ 2,
      HApanic2 == "Very often indeed" ~ 3,
      is.na(HApanic2) ~ NA_real_,
      TRUE ~ NA_real_
    )
  )

# Calculate HADS Anxiety total score for T2
data <- data %>%
  mutate(
    HADS_Anxiety_T2 = rowSums(cbind(HAtense2_score, HAfrightened2_score, HAworry2_score,
                                    HAfeelRelaxed2_score, HAbutterflies2_score, HArestless2_score, HApanic2_score),
                              na.rm = TRUE)
  )

# Recode HADS Anxiety items for T3
data <- data %>%
  mutate(
    HAtense3_score = case_when(
      HAtense3 == "Not at all" ~ 0,
      HAtense3 == "From time to time, occasionally" ~ 1,
      HAtense3 == "A lot of the time" ~ 2,
      HAtense3 == "Most of the time" ~ 3,
      is.na(HAtense3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HAfrightened3_score = case_when(
      HAfrightened3 == "Not at all" ~ 0,
      HAfrightened3 == "A little, but it doesn't worry me" ~ 1,
      HAfrightened3 == "Yes, but not too badly" ~ 2,
      HAfrightened3 == "Very definitely and quite badly" ~ 3,
      is.na(HAfrightened3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HAworry3_score = case_when(
      HAworry3 == "Only occasionally" ~ 0,
      HAworry3 == "From time to time, but not too often" ~ 1,
      HAworry3 == "A lot of the time" ~ 2,
      HAworry3 == "A great deal of the time" ~ 3,
      is.na(HAworry3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HAfeelRelaxed3_score = case_when(
      HAfeelRelaxed3 == "Definitely" ~ 0,
      HAfeelRelaxed3 == "Usually" ~ 1,
      HAfeelRelaxed3 == "Not often" ~ 2,
      HAfeelRelaxed3 == "Not at all" ~ 3,
      is.na(HAfeelRelaxed3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HAbutterflies3_score = case_when(
      HAbutterflies3 == "Not at all" ~ 0,
      HAbutterflies3 == "Occasionally" ~ 1,
      HAbutterflies3 == "Quite often" ~ 2,
      HAbutterflies3 == "Very often" ~ 3,
      is.na(HAbutterflies3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HArestless3_score = case_when(
      HArestless3 == "Not at all" ~ 0,
      HArestless3 == "Not very much" ~ 1,
      HArestless3 == "Quite a lot" ~ 2,
      HArestless3 == "Very much indeed" ~ 3,
      is.na(HArestless3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HApanic3_score = case_when(
      HApanic3 == "Not at all" ~ 0,
      HApanic3 == "Not very often" ~ 1,
      HApanic3 == "Quite often" ~ 2,
      HApanic3 == "Very often indeed" ~ 3,
      is.na(HApanic3) ~ NA_real_,
      TRUE ~ NA_real_
    )
  )

# Calculate HADS Anxiety total score for T3
data <- data %>%
  mutate(
    HADS_Anxiety_T3 = rowSums(cbind(HAtense3_score, HAfrightened3_score, HAworry3_score,
                                    HAfeelRelaxed3_score, HAbutterflies3_score, HArestless3_score, HApanic3_score),
                              na.rm = TRUE)
  )

# Recode HADS Depression items for T1
data <- data %>%
  mutate(
    HDenjoy1_score = case_when(
      HDenjoy1 == "Definitely as much" ~ 0,
      HDenjoy1 == "Not quite so much" ~ 1,
      HDenjoy1 == "Only a little" ~ 2,
      HDenjoy1 == "Hardly at all" ~ 3,
      is.na(HDenjoy1) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDlaugh1_score = case_when(
      HDlaugh1 == "As much as I always could" ~ 0,
      HDlaugh1 == "Not quite so much now" ~ 1,
      HDlaugh1 == "Definitely not so much now" ~ 2,
      HDlaugh1 == "Hardly at all" ~ 3,
      is.na(HDlaugh1) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDcheerful1_score = case_when(
      HDcheerful1 == "Most of the time" ~ 0,
      HDcheerful1 == "Sometimes" ~ 1,
      HDcheerful1 == "Not often" ~ 2,
      HDcheerful1 == "Not at all" ~ 3,
      is.na(HDcheerful1) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDslow1_score = case_when(
      HDslow1 == "Not at all" ~ 0,
      HDslow1 == "Sometimes" ~ 1,
      HDslow1 == "Very often" ~ 2,
      HDslow1 == "Nearly all the time" ~ 3,
      is.na(HDslow1) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDappearance1_score = case_when(
      HDappearance1 == "I take just as much care as ever" ~ 0,
      HDappearance1 == "I may not take quite as much care" ~ 1,
      HDappearance1 == "I don't take as much care as I should" ~ 2,
      HDappearance1 == "I definitely don't take as much care" ~ 3,
      is.na(HDappearance1) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDlookFwd1_score = case_when(
      HDlookFwd1 == "As much as I ever did" ~ 0,
      HDlookFwd1 == "Rather less than I used to" ~ 1,
      HDlookFwd1 == "Definitely less than I used to" ~ 2,
      HDlookFwd1 == "Hardly at all" ~ 3,
      is.na(HDlookFwd1) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDbookTV1_score = case_when(
      HDbookTV1 == "Often" ~ 0,
      HDbookTV1 == "Not often" ~ 1,
      HDbookTV1 == "Sometimes" ~ 2,
      HDbookTV1 == "Very seldom" ~ 3,
      is.na(HDbookTV1) ~ NA_real_,
      TRUE ~ NA_real_
    )
  )

# Calculate HADS Depression total score for T1
data <- data %>%
  mutate(
    HADS_Depression_T1 = rowSums(cbind(HDenjoy1_score, HDlaugh1_score, HDcheerful1_score,
                                       HDslow1_score, HDappearance1_score, HDlookFwd1_score, HDbookTV1_score),
                                 na.rm = TRUE)
  )

# Recode HADS Depression items for T2
data <- data %>%
  mutate(
    HDenjoy2_score = case_when(
      HDenjoy2 == "Definitely as much" ~ 0,
      HDenjoy2 == "Not quite so much" ~ 1,
      HDenjoy2 == "Only a little" ~ 2,
      HDenjoy2 == "Hardly at all" ~ 3,
      is.na(HDenjoy2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDlaugh2_score = case_when(
      HDlaugh2 == "As much as I always could" ~ 0,
      HDlaugh2 == "Not quite so much now" ~ 1,
      HDlaugh2 == "Definitely not so much now" ~ 2,
      HDlaugh2 == "Hardly at all" ~ 3,
      is.na(HDlaugh2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDcheerful2_score = case_when(
      HDcheerful2 == "Most of the time" ~ 0,
      HDcheerful2 == "Sometimes" ~ 1,
      HDcheerful2 == "Not often" ~ 2,
      HDcheerful2 == "Not at all" ~ 3,
      is.na(HDcheerful2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDslow2_score = case_when(
      HDslow2 == "Not at all" ~ 0,
      HDslow2 == "Sometimes" ~ 1,
      HDslow2 == "Very often" ~ 2,
      HDslow2 == "Nearly all the time" ~ 3,
      is.na(HDslow2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDappearance2_score = case_when(
      HDappearance2 == "I take just as much care as ever" ~ 0,
      HDappearance2 == "I may not take quite as much care" ~ 1,
      HDappearance2 == "I don't take as much care as I should" ~ 2,
      HDappearance2 == "I definitely don't take as much care" ~ 3,
      is.na(HDappearance2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDlookFwd2_score = case_when(
      HDlookFwd2 == "As much as I ever did" ~ 0,
      HDlookFwd2 == "Rather less than I used to" ~ 1,
      HDlookFwd2 == "Definitely less than I used to" ~ 2,
      HDlookFwd2 == "Hardly at all" ~ 3,
      is.na(HDlookFwd2) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDbookTV2_score = case_when(
      QHDbookTV2 == "Often" ~ 0,
      QHDbookTV2 == "Not often" ~ 1,
      QHDbookTV2 == "Sometimes" ~ 2,
      QHDbookTV2 == "Very seldom" ~ 3,
      is.na(QHDbookTV2) ~ NA_real_,
      TRUE ~ NA_real_
    )
  )

# Calculate HADS Depression total score for T2
data <- data %>%
  mutate(
    HADS_Depression_T2 = rowSums(cbind(HDenjoy2_score, HDlaugh2_score, HDcheerful2_score,
                                       HDslow2_score, HDappearance2_score, HDlookFwd2_score, HDbookTV2_score),
                                 na.rm = TRUE)
  )

# Recode HADS Depression items for T3
data <- data %>%
  mutate(
    HDenjoy3_score = case_when(
      HDenjoy3 == "Definitely as much" ~ 0,
      HDenjoy3 == "Not quite so much" ~ 1,
      HDenjoy3 == "Only a little" ~ 2,
      HDenjoy3 == "Hardly at all" ~ 3,
      is.na(HDenjoy3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDlaugh3_score = case_when(
      HDlaugh3 == "As much as I always could" ~ 0,
      HDlaugh3 == "Not quite so much now" ~ 1,
      HDlaugh3 == "Definitely not so much now" ~ 2,
      HDlaugh3 == "Hardly at all" ~ 3,
      is.na(HDlaugh3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDcheerful3_score = case_when(
      HDcheerful3 == "Most of the time" ~ 0,
      HDcheerful3 == "Sometimes" ~ 1,
      HDcheerful3 == "Not often" ~ 2,
      HDcheerful3 == "Not at all" ~ 3,
      is.na(HDcheerful3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDslow3_score = case_when(
      HDslow3 == "Not at all" ~ 0,
      HDslow3 == "Sometimes" ~ 1,
      HDslow3 == "Very often" ~ 2,
      HDslow3 == "Nearly all the time" ~ 3,
      is.na(HDslow3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDappearance3_score = case_when(
      HDappearance3 == "I take just as much care as ever" ~ 0,
      HDappearance3 == "I may not take quite as much care" ~ 1,
      HDappearance3 == "I don't take as much care as I should" ~ 2,
      HDappearance3 == "I definitely don't take as much care" ~ 3,
      is.na(HDappearance3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDlookFwd3_score = case_when(
      HDlookFwd3 == "As much as I ever did" ~ 0,
      HDlookFwd3 == "Rather less than I used to" ~ 1,
      HDlookFwd3 == "Definitely less than I used to" ~ 2,
      HDlookFwd3 == "Hardly at all" ~ 3,
      is.na(HDlookFwd3) ~ NA_real_,
      TRUE ~ NA_real_
    ),
    HDbookTV3_score = case_when(
      HDbookTV3 == "Often" ~ 0,
      HDbookTV3 == "Not often" ~ 1,
      HDbookTV3 == "Sometimes" ~ 2,
      HDbookTV3 == "Very seldom" ~ 3,
      is.na(HDbookTV3) ~ NA_real_,
      TRUE ~ NA_real_
    )
  )

# Calculate HADS Depression total score for T3
data <- data %>%
  mutate(
    HADS_Depression_T3 = rowSums(cbind(HDenjoy3_score, HDlaugh3_score, HDcheerful3_score,
                                       HDslow3_score, HDappearance3_score, HDlookFwd3_score, HDbookTV3_score),
                                 na.rm = TRUE)
  )

# ===================== LMMs with Group × Timepoint (baseline T1 reference) =====================
# Uses wide columns in `data`: PEG_T*, ISI_Total_T*, FSS_T*, HADS_Anxiety_T*, HADS_Depression_T*

library(lme4)
library(lmerTest)
library(broom.mixed)
library(emmeans)
library(dplyr)
library(tidyr)
library(readr)
library(purrr)

# Base data
dat0 <- data %>%
  dplyr::mutate(
    ParticipantID = as.factor(ParticipantID),
    Group = factor(Group)
  )

# Make "Control" the reference level if present
if ("Control" %in% levels(dat0$Group)) {
  dat0$Group <- stats::relevel(dat0$Group, ref = "Control")
} else if ("control" %in% levels(dat0$Group)) {
  dat0$Group <- stats::relevel(dat0$Group, ref = "control")
}

# Which outcomes to run (based on your column names)
outcome_specs <- list(
  list(name = "PEG",              cols = c(T1 = "PEG_T1",              T2 = "PEG_T2",              T3 = "PEG_T3")),
  list(name = "ISI_Total",        cols = c(T1 = "ISI_Total_T1",        T2 = "ISI_Total_T2",        T3 = "ISI_Total_T3")),
  list(name = "FSS",              cols = c(T1 = "FSS_T1",              T2 = "FSS_T2",              T3 = "FSS_T3")),
  list(name = "HADS_Anxiety",     cols = c(T1 = "HADS_Anxiety_T1",     T2 = "HADS_Anxiety_T2",     T3 = "HADS_Anxiety_T3")),
  list(name = "HADS_Depression",  cols = c(T1 = "HADS_Depression_T1",  T2 = "HADS_Depression_T2",  T3 = "HADS_Depression_T3"))
)

# ---- REPLACE YOUR fit_one_outcome() WITH THIS ----
# ---- DROP-IN REPLACEMENT: fit_one_outcome() with backticked `df` ----
# ---- REALLY FINAL, ROBUST fit_one_outcome() ----
fit_one_outcome <- function(dat, spec) {
  nm   <- spec$name
  cols <- unname(spec$cols)  # c("..._T1","..._T2","..._T3")
  
  # helpers to normalise columns so select() never fails
  normalize_cols <- function(tbl) {
    # df
    if (!"df" %in% names(tbl)) {
      alt <- intersect(c("df.error","df.residual","den.df","Df","DF"), names(tbl))
      if (length(alt)) tbl <- dplyr::rename(tbl, df = !! rlang::sym(alt[1])) else tbl <- dplyr::mutate(tbl, df = NA_real_)
    }
    # std.error
    if (!"std.error" %in% names(tbl)) {
      alt <- intersect(c("SE","Std..Error","Std.Error","Std_Error"), names(tbl))
      if (length(alt)) tbl <- dplyr::rename(tbl, std.error = !! rlang::sym(alt[1]))
    }
    # statistic (t or z)
    if (!"statistic" %in% names(tbl)) {
      alt <- intersect(c("t.ratio","t.value","t","z.ratio","z.value","z"), names(tbl))
      if (length(alt)) tbl <- dplyr::rename(tbl, statistic = !! rlang::sym(alt[1])) else tbl <- dplyr::mutate(tbl, statistic = NA_real_)
    }
    # p.value (compute if missing and we have df + statistic)
    if (!"p.value" %in% names(tbl)) {
      alt <- intersect(c("p.value.","p","P.Value","Pr(>|t|)","Pr..>.t.."), names(tbl))
      if (length(alt)) {
        tbl <- dplyr::rename(tbl, p.value = !! rlang::sym(alt[1]))
      } else {
        if (all(c("statistic","df") %in% names(tbl)) && is.numeric(tbl$statistic) && is.numeric(tbl$df)) {
          tbl <- dplyr::mutate(tbl, p.value = 2*stats::pt(abs(statistic), df, lower.tail = FALSE))
        } else {
          tbl <- dplyr::mutate(tbl, p.value = NA_real_)
        }
      }
    }
    # 95% CI (compute if missing and we have df + std.error)
    if (!"conf.low" %in% names(tbl) || !"conf.high" %in% names(tbl)) {
      if (all(c("estimate","std.error","df") %in% names(tbl)) && is.numeric(tbl$df)) {
        crit <- stats::qt(0.975, tbl$df)
        if (!"conf.low" %in% names(tbl))  tbl <- dplyr::mutate(tbl,  conf.low  = estimate - crit*std.error)
        if (!"conf.high" %in% names(tbl)) tbl <- dplyr::mutate(tbl, conf.high = estimate + crit*std.error)
      } else {
        if (!"conf.low" %in% names(tbl))  tbl <- dplyr::mutate(tbl,  conf.low  = NA_real_)
        if (!"conf.high" %in% names(tbl)) tbl <- dplyr::mutate(tbl, conf.high = NA_real_)
      }
    }
    tbl
  }
  
  # Skip if required columns missing
  if (!all(cols %in% names(dat))) {
    missing_cols <- setdiff(cols, names(dat))
    message(sprintf("[SKIP] %s: missing columns: %s", nm, paste(missing_cols, collapse = ", ")))
    return(NULL)
  }
  
  long_df <- dat %>%
    dplyr::select(ParticipantID, Group, dplyr::all_of(cols)) %>%
    tidyr::pivot_longer(dplyr::all_of(cols), names_to = "WideName", values_to = "value") %>%
    dplyr::mutate(
      Timepoint = dplyr::case_when(
        WideName == spec$cols["T1"] ~ "T1",
        WideName == spec$cols["T2"] ~ "T2",
        WideName == spec$cols["T3"] ~ "T3",
        TRUE ~ NA_character_
      ),
      Timepoint = factor(Timepoint, levels = c("T1","T2","T3"))
    ) %>%
    dplyr::filter(!is.na(value))
  
  # Need ≥2 timepoints and both groups
  if (nrow(long_df) == 0 || dplyr::n_distinct(long_df$Timepoint) < 2 || dplyr::n_distinct(long_df$Group) < 2) {
    message(sprintf("[SKIP] %s: insufficient data across Group × Timepoint.", nm))
    return(NULL)
  }
  
  m <- lme4::lmer(value ~ Group * Timepoint + (1 | ParticipantID), data = long_df, REML = TRUE)
  
  # Fixed effects
  fx <- broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE, conf.level = 0.95) %>%
    normalize_cols() %>%
    dplyr::mutate(
      outcome = nm,
      term_read = dplyr::case_when(
        term == "(Intercept)"                   ~ "Intercept (Control @ T1)",
        term == "GroupExperimental"             ~ "Group: Experimental vs Control",
        term == "TimepointT2"                   ~ "Time: T2 vs T1 (Control)",
        term == "TimepointT3"                   ~ "Time: T3 vs T1 (Control)",
        term == "GroupExperimental:TimepointT2" ~ "Interaction: (Exp−Ctrl) × (T2−T1)",
        term == "GroupExperimental:TimepointT3" ~ "Interaction: (Exp−Ctrl) × (T3−T1)",
        TRUE ~ term
      )
    ) %>%
    dplyr::select(outcome, term = term_read, estimate, std.error, df, statistic, p.value, conf.low, conf.high)
  
  # EMMs & contrasts
  emms <- emmeans::emmeans(m, ~ Group * Timepoint)
  
  within_changes <- emmeans::contrast(
    emms,
    method = list(
      "Change T2−T1 | Control"      = c(-1,  1,  0,  0,  0,  0),
      "Change T3−T1 | Control"      = c(-1,  0,  1,  0,  0,  0),
      "Change T2−T1 | Experimental" = c( 0,  0,  0, -1,  1,  0),
      "Change T3−T1 | Experimental" = c( 0,  0,  0, -1,  0,  1)
    )
  )
  did_T2 <- emmeans::contrast(emms, list("DiD (T2−T1): Exp−Ctrl" = c( 1, -1,  0,  -1,  1,  0)))
  did_T3 <- emmeans::contrast(emms, list("DiD (T3−T1): Exp−Ctrl" = c( 1,  0, -1,  -1,  0,  1)))
  
  within_tbl <- broom::tidy(within_changes, conf.int = TRUE) %>%
    normalize_cols() %>%
    dplyr::mutate(outcome = nm) %>%
    dplyr::select(outcome, contrast, estimate, std.error, df, statistic, p.value, conf.low, conf.high)
  
  did_tbl <- dplyr::bind_rows(
    broom::tidy(did_T2, conf.int = TRUE),
    broom::tidy(did_T3, conf.int = TRUE)
  ) %>%
    normalize_cols() %>%
    dplyr::mutate(outcome = nm) %>%
    dplyr::select(outcome, contrast, estimate, std.error, df, statistic, p.value, conf.low, conf.high)
  
  list(model = m, fixed = fx, within = within_tbl, did = did_tbl)
}

# ----------------------------------------------------------------------
# Now rebuild results (run this part):
fits_list <- purrr::map(outcome_specs, ~fit_one_outcome(dat0, .x)) %>% purrr::compact()
fixed_effects_table <- purrr::list_rbind(purrr::map(fits_list, "fixed"))
change_within_table <- purrr::list_rbind(purrr::map(fits_list, "within"))
change_did_table    <- purrr::list_rbind(purrr::map(fits_list, "did"))

readr::write_csv(fixed_effects_table, "lmm_fixed_effects_table.csv")
readr::write_csv(change_within_table, "lmm_within_group_changes_from_T1.csv")
readr::write_csv(change_did_table,    "lmm_difference_in_differences.csv")

# --------------------------------------------------

# Then re-run these lines:
fits_list <- purrr::map(outcome_specs, ~fit_one_outcome(dat0, .x)) %>% purrr::compact()

fixed_effects_table <- purrr::list_rbind(purrr::map(fits_list, "fixed"))
change_within_table <- purrr::list_rbind(purrr::map(fits_list, "within"))
change_did_table    <- purrr::list_rbind(purrr::map(fits_list, "did"))

readr::write_csv(fixed_effects_table, "lmm_fixed_effects_table.csv")
readr::write_csv(change_within_table, "lmm_within_group_changes_from_T1.csv")
readr::write_csv(change_did_table,    "lmm_difference_in_differences.csv")

# ==============================================================================================


# Calculate difference scores for each outcome in wk 1 then wk 2 
data <- data %>%
  mutate(
    # PEG (Pain)
    PEG_diff_W1 = PEG_T2 - PEG_T1,
    PEG_diff_W2 = PEG_T3 - PEG_T1,
    
    # ISI (Insomnia)
    ISI_diff_W1 = ISI_Total_T2 - ISI_Total_T1,
    ISI_diff_W2 = ISI_Total_T3 - ISI_Total_T1,
    
    # FSS (Fatigue Severity Scale)
    FSS_diff_W1 = FSS_T2 - FSS_T1,
    FSS_diff_W2 = FSS_T3 - FSS_T1,
    
    # HADS Anxiety
    Anxiety_diff_W1 = HADS_Anxiety_T2 - HADS_Anxiety_T1,
    Anxiety_diff_W2 = HADS_Anxiety_T3 - HADS_Anxiety_T1,
    
    # HADS Depression
    Depression_diff_W1 = HADS_Depression_T2 - HADS_Depression_T1,
    Depression_diff_W2 = HADS_Depression_T3 - HADS_Depression_T1
  )

#r wants long format so 
library(tidyr)

# Pivot longer for all difference scores
long_data <- data %>%
  pivot_longer(
    cols = c(PEG_diff_W1, PEG_diff_W2,
             ISI_diff_W1, ISI_diff_W2,
             FSS_diff_W1, FSS_diff_W2,
             Anxiety_diff_W1, Anxiety_diff_W2,
             Depression_diff_W1, Depression_diff_W2),
    names_to = c("Measure", "Timepoint"),
    names_pattern = "(.*)_diff_(W[12])",
    values_to = "DifferenceScore"
  )
#check long data structure
glimpse(long_data)
table(long_data$Measure)
table(long_data$Timepoint)
table(long_data$Group)

#LMM fititng for each measure next 

model_PEG <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                  data = filter(long_data, Measure == "PEG"))

summary(model_PEG)

model_ISI <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                  data = filter(long_data, Measure == "ISI"))

model_FSS <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                  data = filter(long_data, Measure == "FSS"))

model_Anxiety <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                      data = filter(long_data, Measure == "Anxiety"))

model_Depression <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                         data = filter(long_data, Measure == "Depression"))

#post hoc tests
library(emmeans)

# Get estimated marginal means for PEG 
emm_PEG <- emmeans(model_PEG, ~ Group * Timepoint)

# Pairwise comparisons
contrast(emm_PEG, method = "pairwise", adjust = "holm")

# Get estimated marginal means for PEG 
emm_ISI <- emmeans(model_ISI, ~ Group * Timepoint)

# Pairwise comparisons
contrast(emm_ISI, method = "pairwise", adjust = "holm")

# Get estimated marginal means for PEG 
emm_FSS <- emmeans(model_FSS, ~ Group * Timepoint)

# Pairwise comparisons
contrast(emm_FSS, method = "pairwise", adjust = "holm")

# Get estimated marginal means for PEG 
emm_Anxiety <- emmeans(model_Anxiety, ~ Group * Timepoint)

# Pairwise comparisons
contrast(emm_Anxiety, method = "pairwise", adjust = "holm")

# Get estimated marginal means for PEG 
emm_Depression <- emmeans(model_Depression, ~ Group * Timepoint)

# Pairwise comparisons
contrast(emm_Depression, method = "pairwise", adjust = "holm")

plot(emm_PEG)

#ANOVAs for interaction results etc
library (lme4)
library(lmerTest)

# Example for PEG
anova(model_PEG)

# Repeat for other models:
anova(model_ISI)
anova(model_FSS)
anova(model_Anxiety)
anova(model_Depression)

summary(model_FSS)
summary(model_PEG)

library(lmerTest)

# Refit the models
model_PEG <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                  data = filter(long_data, Measure == "PEG"))

model_ISI <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                  data = filter(long_data, Measure == "ISI"))

model_FSS <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                  data = filter(long_data, Measure == "FSS"))

model_Anxiety <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                      data = filter(long_data, Measure == "Anxiety"))

model_Depression <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID),
                         data = filter(long_data, Measure == "Depression"))

anova(model_PEG)
anova(model_ISI)
anova(model_FSS)
anova(model_Anxiety)
anova(model_Depression)



#residual diagnostics tests

install.packages("DHARMa")
install.packages("sjPlot")
install.packages("influence.ME")

# Load libraries
library(ggplot2)
library(lme4)
library(DHARMa)
library(sjPlot)
library(influence.ME)

# Create a list of all models
models <- list(
  PEG = model_PEG,
  ISI = model_ISI,
  FSS = model_FSS,
  Anxiety = model_Anxiety,
  Depression = model_Depression
)

# Run diagnostics for each model
for (measure in names(models)) {
  cat("\n--- Diagnostics for", measure, "---\n")
  
  model <- models[[measure]]
  
  ## 1. QQ Plot (Normality)
  dev.new()
  qqnorm(residuals(model), main = paste("QQ Plot -", measure))
  qqline(residuals(model), col = "red")
  
  ## 2. Residuals vs. Fitted (Homoscedasticity)
  dev.new()
  plot(fitted(model), residuals(model), 
       main = paste("Residuals vs. Fitted -", measure),
       xlab = "Fitted Values", ylab = "Residuals")
  abline(h = 0, col = "red")
  
  ## 3. Random Effects Visualization
  #dev.new()
 # plot_model(model, type = "re", sort.est = TRUE, 
      #       title = paste("Random Effects -", measure))
  
  ## 4. Cook’s Distance (Influence)
  infl <- influence(model, obs = TRUE)
  cooks <- cooks.distance(infl)
  
  dev.new()
  plot(cooks, type = "h", main = paste("Cook's Distance -", measure),
       ylab = "Cook's Distance", xlab = "Observation")
  abline(h = 4/length(cooks), col = "red", lty = 2)  # Common cutoff line
  
  # Optional: Identify potentially influential points
  influential_points <- which(cooks > 4/length(cooks))
  if (length(influential_points) > 0) {
    cat("Potential influential points detected at observations: ", influential_points, "\n")
  } else {
    cat("No influential points detected.\n")
  }
  
  ## 5. DHARMa Overall Diagnostics
  dev.new()
  sim_res <- simulateResiduals(model)
  plot(sim_res, main = paste("DHARMa Residual Diagnostics -", measure))
  
  ## 6. Shapiro-Wilk Test for Normality
  shapiro_result <- shapiro.test(residuals(model))
  print(shapiro_result)
}

#refit models and final diagnostics 
# Load libraries
library(ggplot2)
library(lme4)
library(DHARMa)

# Create a list of all models
models <- list(
  PEG = model_PEG,
  ISI = model_ISI,
  FSS = model_FSS,
  Anxiety = model_Anxiety,
  Depression = model_Depression
)

# Run diagnostics for each model
for (measure in names(models)) {
  cat("\n--- Diagnostics for", measure, "---\n")
  
  model <- models[[measure]]
  
  # 1. QQ Plot (Normality)
  tryCatch({
    dev.new()
    qqnorm(residuals(model), main = paste("QQ Plot -", measure))
    qqline(residuals(model), col = "red")
  }, error = function(e) {
    cat("Error generating QQ plot for", measure, ":", e$message, "\n")
  })
  
  # 2. Residuals vs. Fitted (Homoscedasticity)
  tryCatch({
    dev.new()
    plot(fitted(model), residuals(model),
         main = paste("Residuals vs. Fitted -", measure),
         xlab = "Fitted Values", ylab = "Residuals")
    abline(h = 0, col = "red")
  }, error = function(e) {
    cat("Error generating Residuals vs. Fitted plot for", measure, ":", e$message, "\n")
  })
  
  # 3. Shapiro-Wilk Test for Normality
  tryCatch({
    shapiro_result <- shapiro.test(residuals(model))
    print(shapiro_result)
  }, error = function(e) {
    cat("Error running Shapiro-Wilk test for", measure, ":", e$message, "\n")
  })
  
  # 4. DHARMa Diagnostics
  tryCatch({
    dev.new()
    sim_res <- simulateResiduals(model)
    plot(sim_res, main = paste("DHARMa Residual Diagnostics -", measure))
  }, error = function(e) {
    cat("Error running DHARMa diagnostics for", measure, ":", e$message, "\n")
  })
}

#do the above again but save out the plots 
# Set this to preferred output folder
output_folder <- "Diagnostics_Plots"

# Create the folder if it doesn't exist
if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# Load libraries
library(ggplot2)
library(lme4)
library(DHARMa)

# Create a list of all models
models <- list(
  PEG = model_PEG,
  ISI = model_ISI,
  FSS = model_FSS,
  Anxiety = model_Anxiety,
  Depression = model_Depression
)

# Run diagnostics for each model and save plots
for (measure in names(models)) {
  cat("\n--- Diagnostics for", measure, "---\n")
  
  model <- models[[measure]]
  
  # 1. QQ Plot
  tryCatch({
    png(filename = file.path(output_folder, paste0(measure, "_QQPlot.png")))
    qqnorm(residuals(model), main = paste("QQ Plot -", measure))
    qqline(residuals(model), col = "red")
    dev.off()
  }, error = function(e) {
    cat("Error generating QQ plot for", measure, ":", e$message, "\n")
  })
  
  # 2. Residuals vs. Fitted Plot
  tryCatch({
    png(filename = file.path(output_folder, paste0(measure, "_Residuals_vs_Fitted.png")))
    plot(fitted(model), residuals(model),
         main = paste("Residuals vs. Fitted -", measure),
         xlab = "Fitted Values", ylab = "Residuals")
    abline(h = 0, col = "red")
    dev.off()
  }, error = function(e) {
    cat("Error generating Residuals vs. Fitted plot for", measure, ":", e$message, "\n")
  })
  
  # 3. Shapiro-Wilk Test
  tryCatch({
    shapiro_result <- shapiro.test(residuals(model))
    print(shapiro_result)
  }, error = function(e) {
    cat("Error running Shapiro-Wilk test for", measure, ":", e$message, "\n")
  })
  
  # 4. DHARMa Diagnostics
  tryCatch({
    png(filename = file.path(output_folder, paste0(measure, "_DHARMaDiagnostics.png")))
    sim_res <- simulateResiduals(model)
    plot(sim_res, main = paste("DHARMa Residual Diagnostics -", measure))
    dev.off()
  }, error = function(e) {
    cat("Error running DHARMa diagnostics for", measure, ":", e$message, "\n")
  })
}

#exploratory analysis of freq and enjoyment correlations with outcome measures in exp grp

colnames(long_data)

# Create a dataset with enjoyment and frequency from the original data
enjoyment_freq <- data %>%
  filter(Group == "Experimental") %>%
  select(ParticipantID, EnjoyedListening2, EnjoyedListening3, HowOften2, HowOften3)


# Focus on experimental group only
experimental_data <- long_data %>%
  filter(Group == "Experimental") %>%
  select(ParticipantID, Measure, Timepoint, DifferenceScore,
         EnjoyedListening2, EnjoyedListening3, HowOften2, HowOften3)

# Create a new variable for Enjoyment and Frequency at each timepoint
experimental_data <- experimental_data %>%
  mutate(
    Enjoyment = case_when(
      Timepoint == "W1" ~ EnjoyedListening2,
      Timepoint == "W2" ~ EnjoyedListening3
    ),
    Frequency = case_when(
      Timepoint == "W1" ~ as.numeric(HowOften2),
      Timepoint == "W2" ~ as.numeric(HowOften3)
    )
  )

# For each clinical outcome, run Spearman correlations with Enjoyment and Frequency
outcomes <- unique(experimental_data$Measure)

for (outcome in outcomes) {
  cat("\n--- Spearman Correlation for", outcome, "---\n")
  
  outcome_data <- experimental_data %>% filter(Measure == outcome)
  
  # Correlation with Enjoyment
  cat("Enjoyment:\n")
  print(cor.test(outcome_data$DifferenceScore, outcome_data$Enjoyment, method = "spearman"))
  
  # Correlation with Frequency
  cat("Frequency:\n")
  print(cor.test(outcome_data$DifferenceScore, outcome_data$Frequency, method = "spearman"))
}

#look at the diff scores for the mood tests to see if there's a floor effect
#begin by looking at the baseline mean scores then calculate change and plot

# Mean and SD at baseline (T1)
mean_anx <- mean(long_data$HADS_Anxiety_T1, na.rm = TRUE)
sd_anx <- sd(long_data$HADS_Anxiety_T1, na.rm = TRUE)

mean_dep <- mean(long_data$HADS_Depression_T1, na.rm = TRUE)
sd_dep <- sd(long_data$HADS_Depression_T1, na.rm = TRUE)

cat("HADS Anxiety T1: M =", mean_anx, ", SD =", sd_anx, "\n")
cat("HADS Depression T1: M =", mean_dep, ", SD =", sd_dep, "\n")

#plot histo spread
library(ggplot2)

# Histogram for anxiety at T1
ggplot(long_data, aes(x = HADS_Anxiety_T1)) +
  geom_histogram(binwidth = 1, fill = "lightblue", color = "black") +
  geom_vline(xintercept = 8, linetype = "dashed", color = "red") +
  labs(title = "Baseline HADS Anxiety (T1)", x = "Score", y = "Count")

# Histogram for depression at T1
ggplot(long_data, aes(x = HADS_Depression_T1)) +
  geom_histogram(binwidth = 1, fill = "lightpink", color = "black") +
  geom_vline(xintercept = 8, linetype = "dashed", color = "red") +
  labs(title = "Baseline HADS Depression (T1)", x = "Score", y = "Count")

#diff scores
long_data$AnxietyChange <- long_data$HADS_Anxiety_T3 - long_data$HADS_Anxiety_T1
long_data$DepressionChange <- long_data$HADS_Depression_T3 - long_data$HADS_Depression_T1

# Descriptives
mean_diff_anx <- mean(long_data$AnxietyChange, na.rm = TRUE)
sd_diff_anx <- sd(long_data$AnxietyChange, na.rm = TRUE)

mean_diff_dep <- mean(long_data$DepressionChange, na.rm = TRUE)
sd_diff_dep <- sd(long_data$DepressionChange, na.rm = TRUE)

cat("Anxiety Change (T3 - T1): M =", mean_diff_anx, ", SD =", sd_diff_anx, "\n")
cat("Depression Change (T3 - T1): M =", mean_diff_dep, ", SD =", sd_diff_dep, "\n")

# Plot the change scores
ggplot(long_data, aes(x = AnxietyChange)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Change in Anxiety (T3 - T1)", x = "Change Score", y = "Count")

ggplot(long_data, aes(x = DepressionChange)) +
  geom_histogram(binwidth = 1, fill = "salmon", color = "black") +
  labs(title = "Change in Depression (T3 - T1)", x = "Change Score", y = "Count")

# Count of clinical levels
table(cut(long_data$HADS_Anxiety_T1,
          breaks = c(-Inf, 7, 10, Inf),
          labels = c("Normal", "Borderline", "Clinical")))

table(cut(long_data$HADS_Depression_T1,
          breaks = c(-Inf, 7, 10, Inf),
          labels = c("Normal", "Borderline", "Clinical")))
#count of clinical categories 
baseline_data <- long_data[!duplicated(long_data$ParticipantID), ]
table(cut(baseline_data$HADS_Anxiety_T1,
          breaks = c(-Inf, 7, 10, Inf),
          labels = c("Normal", "Borderline", "Clinical")))

table(cut(baseline_data$HADS_Depression_T1,
          breaks = c(-Inf, 7, 10, Inf),
          labels = c("Normal", "Borderline", "Clinical")))

table(table(long_data$ParticipantID))


#new long data to test out fuzzy matching 
# Install if necessary
install.packages("stringdist")
library(stringdist)
library(dplyr)

# Step 0: Create a copy of the original data
long_data_corrected <- long_data

# Step 1: Extract Unique Codes from T1
t1_codes <- unique(long_data_corrected$UniqueCode1)

# Step 2: Fuzzy matching function
fuzzy_match_codes <- function(code_set, reference_set, max_distance = 0.1) {
  sapply(code_set, function(code) {
    if (is.na(code)) return(NA)
    distances <- stringdist::stringdist(code, reference_set, method = "jw")
    if (min(distances) <= max_distance) {
      return(reference_set[which.min(distances)])
    } else {
      return(NA)
    }
  })
}

# Step 3: Apply fuzzy matching to T2 and T3 codes
long_data_corrected$CorrectedCode2 <- fuzzy_match_codes(long_data_corrected$UniqueCode2, t1_codes)
long_data_corrected$CorrectedCode3 <- fuzzy_match_codes(long_data_corrected$UniqueCode3, t1_codes)

# Step 4: Replace UniqueCode2/3 values where corrected versions exist
long_data_corrected$UniqueCode2[!is.na(long_data_corrected$CorrectedCode2)] <- 
  long_data_corrected$CorrectedCode2[!is.na(long_data_corrected$CorrectedCode2)]

long_data_corrected$UniqueCode3[!is.na(long_data_corrected$CorrectedCode3)] <- 
  long_data_corrected$CorrectedCode3[!is.na(long_data_corrected$CorrectedCode3)]

# Step 5: Define ParticipantID (based on any available unique code)
long_data_corrected$ParticipantID <- dplyr::coalesce(
  long_data_corrected$UniqueCode1,
  long_data_corrected$UniqueCode2,
  long_data_corrected$UniqueCode3
)

# Step 6 (optional): Check how many extra matches were recovered
cat("Corrected matches at T2:", sum(!is.na(long_data_corrected$CorrectedCode2)), "\n")
cat("Corrected matches at T3:", sum(!is.na(long_data_corrected$CorrectedCode3)), "\n")

# How many unique participants overall?
cat("Total unique participants:", length(unique(long_data_corrected$ParticipantID)), "\n")

# Number of participants who completed each timepoint
completed_T1 <- long_data_corrected %>%
  filter(!is.na(UniqueCode1)) %>%
  distinct(ParticipantID)

completed_T2 <- long_data_corrected %>%
  filter(!is.na(UniqueCode2)) %>%
  distinct(ParticipantID)

completed_T3 <- long_data_corrected %>%
  filter(!is.na(UniqueCode3)) %>%
  distinct(ParticipantID)

# Print results
cat("Participants with T1 data:", nrow(completed_T1), "\n")
cat("Participants with T2 data:", nrow(completed_T2), "\n")
cat("Participants with T3 data:", nrow(completed_T3), "\n")

# Who completed all three?
completed_all <- intersect(intersect(completed_T1$ParticipantID, completed_T2$ParticipantID), completed_T3$ParticipantID)
cat("Participants who completed all 3 timepoints:", length(completed_all), "\n")

head(long_data, 20)
colnames(long_data)


table(table(paste(long_data$ParticipantID, long_data$Timepoint)))

#AT THUS POINT I REALISE THE LONG DATA IS DUPLICATED AND WRONG, REDO ALL ANALYSIS HERE
#clean the long data it has duplicates 
long_data_clean <- long_data %>%
  group_by(ParticipantID, Timepoint) %>%
  slice(1) %>%
  ungroup()
table(table(paste(long_data_clean$ParticipantID, long_data_clean$Timepoint)))


table(long_data_clean$Measure)
filter(long_data_clean, Measure == "ISI") %>% summary()

#new code for the clean data for longdata
long_data_clean <- long_data %>%
  filter(!is.na(DifferenceScore)) %>%  # Only keep valid scores
  group_by(ParticipantID, Timepoint, Measure) %>%
  slice(1) %>%
  ungroup()
table(long_data_clean$Measure)

#starting it all again here becuase that clean data was wrong so this attempts to fix it before doing the prereg
#LMMs to get the stuff we need for the table in the final draft 

library(dplyr)
library(tidyr)

# Build change-from-baseline (W1 = T2-T1, W2 = T3-T1) in wide form
wide_diffs <- data %>%
  transmute(
    ParticipantID,
    Group,
    PEG_W1        = PEG_T2 - PEG_T1,
    PEG_W2        = PEG_T3 - PEG_T1,
    ISI_W1        = ISI_Total_T2 - ISI_Total_T1,
    ISI_W2        = ISI_Total_T3 - ISI_Total_T1,
    FSS_W1        = FSS_T2 - FSS_T1,
    FSS_W2        = FSS_T3 - FSS_T1,
    Anxiety_W1    = HADS_Anxiety_T2 - HADS_Anxiety_T1,
    Anxiety_W2    = HADS_Anxiety_T3 - HADS_Anxiety_T1,
    Depression_W1 = HADS_Depression_T2 - HADS_Depression_T1,
    Depression_W2 = HADS_Depression_T3 - HADS_Depression_T1
  )

# Pivot to long: one row per Participant × Measure × Timepoint
long_data <- wide_diffs %>%
  pivot_longer(
    cols = -c(ParticipantID, Group),
    names_to = c("Measure","Timepoint"),
    names_sep = "_",
    values_to = "DifferenceScore"
  ) %>%
  mutate(
    Timepoint = factor(Timepoint, levels = c("W1","W2")),
    Measure   = factor(Measure, levels = c("PEG","ISI","FSS","Anxiety","Depression"))
  )

# Clean version: drop rows where the diff score is missing
long_data_clean <- long_data %>% filter(!is.na(DifferenceScore))

# Sanity checks
long_data %>% count(Measure, Timepoint)
long_data_clean %>% count(Measure, Timepoint)
#new prereg lmm and results for table 
# ==== PREREG LMMs — tidy tables with estimates, dfs, p, CIs ====
library(lmerTest)
library(emmeans)
library(dplyr)
library(tidyr)
library(purrr)
library(tibble)
library(broom)
library(readr)

# Use your already-built long_data_clean (Δ scores; W1 = T2−T1, W2 = T3−T1)
df0 <- long_data_clean %>%
  mutate(
    ParticipantID = factor(ParticipantID),
    Group = factor(Group),
    Timepoint = factor(Timepoint, levels = c("W1","W2"))
  )

# Make Control the reference
if ("Control" %in% levels(df0$Group)) df0$Group <- stats::relevel(df0$Group, "Control")

# Type-III ANOVA needs sum-to-zero contrasts (helps with imbalance too)
options(contrasts = c("contr.sum","contr.poly"))

fit_meas <- function(meas){
  dd <- df0 %>% dplyr::filter(Measure == meas) %>% tidyr::drop_na(DifferenceScore)
  if (nrow(dd) < 6 || dplyr::n_distinct(dd$Group) < 2 || dplyr::n_distinct(dd$Timepoint) < 2) {
    message(sprintf("[SKIP] %s: insufficient data.", meas)); return(NULL)
  }
  
  m <- lmer(DifferenceScore ~ Group * Timepoint + (1 | ParticipantID), data = dd, REML = TRUE)
  
  # Fixed effects (Satterthwaite df/p via lmerTest's summary method)
  sm <- as.data.frame(summary(m)$coefficients) %>% tibble::rownames_to_column("raw_term")
  fixed_tbl <- sm %>%
    dplyr::transmute(
      outcome = meas,
      term = dplyr::case_when(
        raw_term == "(Intercept)"                   ~ "Intercept (Control @ W1: ΔT2−T1)",
        raw_term == "GroupExperimental"             ~ "Group (Exp−Ctrl) @ W1",
        raw_term == "TimepointW2"                   ~ "Time (W2−W1) in Control",
        raw_term == "GroupExperimental:TimepointW2" ~ "Interaction: (Exp−Ctrl) × (W2−W1)",
        TRUE ~ raw_term
      ),
      estimate  = `Estimate`,
      std.error = `Std. Error`,
      df        = `df`,
      statistic = `t value`,
      p.value   = `Pr(>|t|)`,
      conf.low  = estimate - 1.96 * std.error,
      conf.high = estimate + 1.96 * std.error
    )
  
  # Type-III ANOVA (Satterthwaite) — IMPORTANT: no namespace prefix here
  a3 <- anova(m, type = 3) %>% as.data.frame() %>% tibble::rownames_to_column("effect")
  type3_tbl <- a3 %>% dplyr::transmute(
    outcome = meas, effect,
    num.df = `NumDF`, den.df = `DenDF`, F = `F value`, p.value = `Pr(>F)`
  )
  
  # replace just this block inside fit_meas(), nothing else
  em_grp <- emmeans::emmeans(m, ~ Group)      # levels in order: Control, Experimental
  grp_main_tbl <- emmeans::contrast(
    em_grp,
    method = list("Exp − Ctrl (avg W1,W2)" = c(-1, 1))  # Experimental − Control
  ) |>
    broom::tidy(conf.int = TRUE) |>
    dplyr::transmute(
      outcome = meas,
      contrast = contrast,
      estimate, std.error, df, statistic, p.value, conf.low, conf.high
    )
  
  # Within-group W2 − W1 (exploratory)
  emms <- emmeans::emmeans(m, ~ Group * Timepoint)  # Ctrl.W1, Ctrl.W2, Exp.W1, Exp.W2
  within_tbl <- emmeans::contrast(
    emms,
    method = list("Control: W2−W1"      = c(-1, 1,  0, 0),
                  "Experimental: W2−W1" = c( 0, 0, -1, 1))
  ) %>%
    broom::tidy(conf.int = TRUE) %>%
    dplyr::transmute(outcome = meas, contrast, estimate, std.error, df, statistic, p.value, conf.low, conf.high)
  
  list(model = m, fixed = fixed_tbl, type3 = type3_tbl,
       group_main = grp_main_tbl, within = within_tbl)
}

# Fit all five prereg outcomes
measures <- c("PEG","ISI","FSS","Anxiety","Depression")
fits <- purrr::map(measures, fit_meas) %>% purrr::compact()

# Bind the pieces into data frames
fixed_effects_table <- purrr::list_rbind(purrr::map(fits, "fixed"))
type3_tests_table   <- purrr::list_rbind(purrr::map(fits, "type3"))
group_main_table    <- purrr::list_rbind(purrr::map(fits, "group_main"))
within_simple_table <- purrr::list_rbind(purrr::map(fits, "within"))

dplyr::filter(type3_tests_table, outcome == "FSS")
dplyr::filter(group_main_table,  outcome == "FSS")


# Optional: save for your manuscript tables
readr::write_csv(fixed_effects_table, "LMM_fixed_effects_preregistered.csv")
readr::write_csv(type3_tests_table,   "LMM_type3_tests_preregistered.csv")
readr::write_csv(group_main_table,    "LMM_group_main_preregistered.csv")
readr::write_csv(within_simple_table, "LMM_within_W2_minus_W1_preregistered.csv")

#residual diagnostics 






#floor effect checks

colnames(long_data_clean)
head(long_data_clean)
unique(long_data_clean$Timepoint)
unique(long_data_clean$Measure)
long_data_clean %>%
  filter(Measure %in% c("Anxiety", "Depression")) %>%
  sample_n(10)
#baseline anxiety and depression scores 
library(ggplot2)
library(tidyverse)

# Extract Anxiety and Depression Week 1 scores
baseline_plot_data <- long_data_clean %>%
  dplyr::select(ParticipantID, HADS_Anxiety_T1, HADS_Depression_T1) %>%
  pivot_longer(
    cols = c(HADS_Anxiety_T1, HADS_Depression_T1),
    names_to = "Measure",
    values_to = "Score"
  ) %>%
  mutate(
    Measure = recode(Measure,
                     HADS_Anxiety_T1 = "Anxiety",
                     HADS_Depression_T1 = "Depression")
  )

ggplot(baseline_plot_data, aes(x = Score)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  facet_wrap(~Measure, scales = "free_y") +
  labs(
    title = "Distribution of Baseline HADS Scores (Week 1)",
    x = "Total Score",
    y = "Count"
  ) +
  theme_minimal()
# Add mean and clinical threshold lines
baseline_plot_data %>%
  ggplot(aes(x = Score)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  geom_vline(data = baseline_plot_data %>% group_by(Measure) %>% 
               summarise(mean_score = mean(Score, na.rm = TRUE)),
             aes(xintercept = mean_score), 
             linetype = "dashed", color = "red", linewidth = 1) +
  geom_vline(xintercept = 8, linetype = "dashed", color = "black") +  # Clinical threshold
  facet_wrap(~Measure, scales = "free_y") +
  labs(
    title = "Distribution of Baseline HADS Scores (Week 1)",
    x = "Total Score",
    y = "Count"
  ) +
  theme_minimal()
# Calculate percentages above and below cutoff (>= 8)
baseline_summary <- baseline_plot_data %>%
  mutate(AboveCutoff = Score >= 8) %>%
  group_by(Measure, AboveCutoff) %>%
  summarise(N = n(), .groups = "drop") %>%
  group_by(Measure) %>%
  mutate(Percent = round(100 * N / sum(N), 1))

baseline_summary

#get the diff score data
long_data_clean <- as_tibble(long_data_clean)

diff_data <- long_data_clean %>%
  dplyr::select(ParticipantID,
                HADS_Anxiety_T1, HADS_Anxiety_T2, HADS_Anxiety_T3,
                HADS_Depression_T1, HADS_Depression_T2, HADS_Depression_T3) %>%
  mutate(
    Anxiety_Change_T2_T1 = HADS_Anxiety_T2 - HADS_Anxiety_T1,
    Anxiety_Change_T3_T1 = HADS_Anxiety_T3 - HADS_Anxiety_T1,
    Depression_Change_T2_T1 = HADS_Depression_T2 - HADS_Depression_T1,
    Depression_Change_T3_T1 = HADS_Depression_T3 - HADS_Depression_T1
  )

library(ggplot2)
library(tidyr)

diff_long <- diff_data %>%
  pivot_longer(
    cols = starts_with("Anxiety_Change") | starts_with("Depression_Change"),
    names_to = "Change_Type",
    values_to = "Score"
  )

diff_long <- diff_long %>%
  mutate(
    Change_Type = recode(Change_Type,
                         "Anxiety_Change_T2_T1" = "Anxiety (T2 − T1)",
                         "Anxiety_Change_T3_T1" = "Anxiety (T3 − T1)",
                         "Depression_Change_T2_T1" = "Depression (T2 − T1)",
                         "Depression_Change_T3_T1" = "Depression (T3 − T1)"
    )
  )

ggplot(diff_long, aes(x = Score)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  facet_wrap(~Change_Type, scales = "free_y") +
  labs(
    title = "Distribution of Change Scores in HADS Anxiety and Depression",
    x = "Score Change (Post - Baseline)",
    y = "Count"
  ) +
  theme_minimal()
library(dplyr)

# Compute group means
means <- diff_long %>%
  group_by(Change_Type) %>%
  summarize(mean_score = mean(Score, na.rm = TRUE))
#add denstiy curve to the plots
ggplot(diff_long, aes(x = Score)) +
  geom_histogram(aes(y = ..density..), binwidth = 1, fill = "skyblue", color = "black", alpha = 0.7) +
  geom_density(color = "darkblue", size = 1) +
  geom_vline(data = means, aes(xintercept = mean_score), color = "red", linetype = "dashed", size = 1) +
  facet_wrap(~Change_Type, scales = "free_y") +
  labs(
    title = "Distribution of HADS Change Scores",
    x = "Score Change (Post - Baseline)",
    y = "Density"
  ) +
  theme_minimal()
# Calculate improvement percentages
improvement_stats <- diff_long %>%
  group_by(Change_Type) %>%
  summarize(
    N = n(),
    Improved = sum(Score < 0, na.rm = TRUE),
    Percent_Improved = round(100 * Improved / N, 1)
  )
print(improvement_stats)

diff_data %>%
  summarise(
    M_Anx_T2_T1 = mean(Anxiety_Change_T2_T1, na.rm = TRUE),
    SD_Anx_T2_T1 = sd(Anxiety_Change_T2_T1, na.rm = TRUE),
    M_Dep_T2_T1 = mean(Depression_Change_T2_T1, na.rm = TRUE),
    SD_Dep_T2_T1 = sd(Depression_Change_T2_T1, na.rm = TRUE)
  )

#exploratory analysis on freq and enjoyment attempt
library(dplyr)
exp_data <- exp_data %>%
  mutate(
    HowOften2_num = case_when(
      HowOften2 == "More frequently than once a day" ~ 3,
      HowOften2 == "Once every day this week" ~ 2,
      HowOften2 == "Less frequently than once a day" ~ 1,
      HowOften2 == "I have not been able to listen to them at all this week" ~ 0,
      TRUE ~ NA_real_
    ),
    HowOften3_num = case_when(
      HowOften3 == "More frequently than once a day" ~ 3,
      HowOften3 == "Once every day this week" ~ 2,
      HowOften3 == "Less frequently than once a day" ~ 1,
      HowOften3 == "I have not been able to listen to them at all this week" ~ 0,
      TRUE ~ NA_real_
    )
  )

exp_data <- exp_data %>%
  mutate(
    EnjoyedListening2_num = case_when(
      EnjoyedListening2 == "Yes, all of them" ~ 2,
      EnjoyedListening2 == "Yes, some of them" ~ 1,
      EnjoyedListening2 == "No, none of them" ~ 0,
      TRUE ~ NA_real_
    ),
    EnjoyedListening3_num = case_when(
      EnjoyedListening3 == "Yes, all of them" ~ 2,
      EnjoyedListening3 == "Yes, some of them" ~ 1,
      EnjoyedListening3 == "No, none of them" ~ 0,
      TRUE ~ NA_real_
    )
  )

colnames(exp_data)
#add missing change scores for clincial outcomes 
library(dplyr)

exp_data <- exp_data %>%
  mutate(
    # Fatigue Severity Scale change scores
    FSS_change_T2_T1 = FSS_T2 - FSS_T1,
    
    # PEG (Pain Interference) change scores
    PEG_change_T2_T1 = PEG_T2 - PEG_T1,
    
    # ISI (Insomnia Severity Index) change scores
    ISI_change_T2_T1 = ISI_Total_T2 - ISI_Total_T1,
    ISI_change_T3_T1 = ISI_Total_T3 - ISI_Total_T1
  )

exp_data <- exp_data %>%
  mutate(
    FSS_change_T2_T1 = ifelse(is.na(FSS_change_T2_T1), FSS_T2 - FSS_T1, FSS_change_T2_T1),
    PEG_change_T2_T1 = ifelse(is.na(PEG_change_T2_T1), PEG_T2 - PEG_T1, PEG_change_T2_T1),
    ISI_change_T2_T1 = ISI_Total_T2 - ISI_Total_T1,
    ISI_change_T3_T1 = ISI_Total_T3 - ISI_Total_T1
  ) 
#T2 correlation 
cor_T2 <- exp_data %>%
  dplyr::select(
    HowOften2_num, EnjoyedListening2_num,
    AnxietyChange, DepressionChange,
    ISI_change_T2_T1, FSS_change_T2_T1, PEG_change_T2_T1
  ) %>%
  cor(use = "pairwise.complete.obs")

print(round(cor_T2, 2))

#T3 correlation 
cor_T3 <- exp_data %>%
  dplyr::select(
    HowOften3_num, EnjoyedListening3_num,
    AnxietyChange, DepressionChange,
    ISI_change_T3_T1, FSS_change_T3_T1, PEG_change_T3_T1
  ) %>%
  cor(use = "pairwise.complete.obs")
print(round(cor_T3, 2))


colnames(long_data_clean)

#new exploratory data attempt 
# --- Clean exploratory correlations (T3 engagement vs ΔT3−T1) ---

library(dplyr)
library(tidyr)
library(purrr)
library(broom)
library(tibble)

# 1) Experimental only, code T3 engagement properly
exp <- data %>%
  filter(Group == "Experimental") %>%
  transmute(
    ParticipantID,
    # Engagement at T3 (end of intervention)
    HowOften3_num = case_when(
      HowOften3 == "More frequently than once a day" ~ 3,
      HowOften3 == "Once every day this week"        ~ 2,
      HowOften3 == "Less frequently than once a day" ~ 1,
      HowOften3 == "I have not been able to listen to them at all this week" ~ 0,
      TRUE ~ NA_real_
    ),
    EnjoyedListening3_num = case_when(
      EnjoyedListening3 == "Yes, all of them" ~ 2,
      EnjoyedListening3 == "Yes, some of them" ~ 1,
      EnjoyedListening3 == "No, none of them"  ~ 0,
      TRUE ~ NA_real_
    ),
    # Δ(T3 − T1) outcomes (negative = improvement)
    PEG_change_T3_T1        = PEG_T3 - PEG_T1,
    FSS_change_T3_T1        = FSS_T3 - FSS_T1,
    ISI_change_T3_T1        = ISI_Total_T3 - ISI_Total_T1,
    Anxiety_change_T3_T1    = HADS_Anxiety_T3 - HADS_Anxiety_T1,
    Depression_change_T3_T1 = HADS_Depression_T3 - HADS_Depression_T1
  )

# 2) Tidy correlation tests: 5 outcomes × 2 predictors = 10 tests
outcome_vars   <- c("PEG_change_T3_T1","FSS_change_T3_T1","ISI_change_T3_T1",
                    "Anxiety_change_T3_T1","Depression_change_T3_T1")
predictor_vars <- c("HowOften3_num","EnjoyedListening3_num")

corr_results <- purrr::map_dfr(outcome_vars, function(y) {
  purrr::map_dfr(predictor_vars, function(x) {
    # <-— FIX: use base subsetting by names to dodge all_of()
    df <- exp[, c(x, y)]
    # drop rows with any NA in the two selected columns
    df <- df[stats::complete.cases(df), , drop = FALSE]
    n <- nrow(df)
    if (is.null(n) || n < 6) {
      return(tibble(
        outcome  = y, predictor = x, n = n,
        r = NA_real_, p = NA_real_, conf.low = NA_real_, conf.high = NA_real_
      ))
    }
    ct <- stats::cor.test(df[[x]], df[[y]], method = "pearson")
    tibble(
      outcome   = y,
      predictor = x,
      n         = n,
      r         = unname(ct$estimate),
      p         = ct$p.value,
      conf.low  = unname(ct$conf.int[1]),
      conf.high = unname(ct$conf.int[2])
    )
  })
})

# Holm–Bonferroni across the 10 tests + add df for reporting
corr_results <- corr_results %>%
  mutate(
    p_holm = p.adjust(p, method = "holm"),
    df     = n - 2
  ) %>%
  mutate(
    outcome = recode(outcome,
                     PEG_change_T3_T1        = "PEG (T3−T1)",
                     FSS_change_T3_T1        = "FSS (T3−T1)",
                     ISI_change_T3_T1        = "ISI (T3−T1)",
                     Anxiety_change_T3_T1    = "Anxiety (T3−T1)",
                     Depression_change_T3_T1 = "Depression (T3−T1)"
    ),
    predictor = recode(predictor,
                       HowOften3_num          = "Listening frequency (T3)",
                       EnjoyedListening3_num  = "Enjoyment (T3)"
    )
  ) %>%
  arrange(outcome, predictor)

corr_results
# Optionally save:
# readr::write_csv(corr_results, "Exploratory_Correlations_T3.csv")

# overall analytic N used for LMMs (any outcome, any timepoint)
dplyr::n_distinct(long_data$ParticipantID)

# confirm on the final LMM dataset you used
dplyr::n_distinct(long_data_clean$ParticipantID)

# per-group counts (overall)
long_data %>% dplyr::distinct(ParticipantID, Group) %>% dplyr::count(Group)

# per-model/timepoint row counts (already matched your printouts)
long_data_clean %>% dplyr::count(Measure, Timepoint)

# IDs that appear with more than one group label
long_data %>%
  dplyr::distinct(ParticipantID, Group) %>%
  dplyr::count(ParticipantID) %>%
  dplyr::filter(n > 1)

# Any rows with missing Group?
long_data %>% dplyr::filter(is.na(Group)) %>% nrow()

# Baseline-defined group per participant
group_by_id <- data %>% 
  dplyr::select(ParticipantID, Group) %>%
  dplyr::distinct(ParticipantID, .keep_all = TRUE)

group_by_id %>% dplyr::count(Group)


