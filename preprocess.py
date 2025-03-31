#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:52:45 2025

@author: Alaina

Drop unnecessary columns, missing values, get number of trials, and perform
an 80/ 20 train test split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

window_size = 5
random_state = 42

filepath = f"group_R_features_slide_wlen{window_size}.csv"
data = pd.read_csv(filepath,index_col=0)

# trials calculations - pre dropping
trial_calc = data.copy()

# make time diff and new trial cols
trial_calc["time_diff"] = trial_calc["relative_time"].diff()
trial_calc["new_trial"] = (trial_calc["time_diff"].isna()) | (trial_calc["time_diff"].abs() > .25)
# make ID col
trial_calc["trial_id"] = trial_calc["new_trial"].cumsum()



# drop same rows from trial_calc as data, we have unique ID to serve as way to ID trials now


#print("length of full data before dropping offset", len(data))
# drop rows where event is mw_offset now so changes are reflected in entire dataset
data = data[data["label"] != "MW_offset"]
trial_calc = trial_calc[trial_calc["label"] != "MW_offset"]
#print("length of full data after dropping offset", len(data))
#print(data.isna().sum())



# handle missing values - drop horizontal_sacc then drop the nas
data.drop(columns=["horizontal_sacc", "blink_num", "blink_dur", "blink_freq"], inplace=True)
trial_calc.drop(columns=["horizontal_sacc", "blink_num", "blink_dur", "blink_freq"], inplace=True)




# feature extraction - create X and y for splitting
# labels is included in features at this stage, but dropped later (same with page)

features = ["fix_num","label", "norm_fix_word_num", "norm_in_word_reg",
            "norm_out_word_reg", "zscored_zipf_fixdur_corr", "zipf_fixdur_corr",
            "zscored_word_length_fixdur_corr","norm_total_viewing", "fix_dispersion",
            "weighted_vergence", "norm_sacc_num",
            "sacc_length","norm_pupil", "page", "relative_time"]

# keep subject id in X for LOGOCV

X_features = ["fix_num","label", "norm_fix_word_num", "norm_in_word_reg",
            "norm_out_word_reg", "zscored_zipf_fixdur_corr", "zipf_fixdur_corr",
            "zscored_word_length_fixdur_corr","norm_total_viewing", "fix_dispersion",
            "weighted_vergence", "norm_sacc_num",
            "sacc_length","norm_pupil", "page", "relative_time", "sub_id"]


data.dropna(subset = features, inplace=True)
trial_calc.dropna(subset = features, inplace=True)



print(data["label"].unique())

"""
- don't count trials where all corresponding rows were dropped - have trial ID then 
count unique ID after dropping (segment into different sets for each event first
                                to have counts of trials for each)
- save num trials to a csv so I can load into onset v self report instead of recalculating
"""



sr_trials = trial_calc[trial_calc["label"] == "self_report"]
mw_trials = trial_calc[trial_calc["label"] == "MW_onset"]
ctl_trials = trial_calc[trial_calc["label"] == "control"]

sr_trial_count = sr_trials["trial_id"].nunique()
mw_trial_count = mw_trials["trial_id"].nunique()
ctl_trial_count = ctl_trials["trial_id"].nunique()


print(sr_trial_count, " sr trials")
print(mw_trial_count, " mw trials")
print(ctl_trial_count, " ctl trials")

trial_count_df = pd.DataFrame({
    "MW_trials": [mw_trial_count],
    "sr_trials": [sr_trial_count],
    "ctl_trials": [ctl_trial_count]
    })

#pd.set_option("display.max_columns", None)
#trial_calc.head()
# for each of these groups, indicate new trial when 


"""
# old, broken. This strategy doesn't account for rows dropped due to 
# missing values
sr_trial_calc = trial_calc[trial_calc["label"] == "self_report"]
mw_trial_calc = trial_calc[trial_calc["label"] == "MW_onset"]
ctl_trial_calc = trial_calc[(trial_calc["label"] == "control")]

sr_trial_calc = sr_trial_calc.copy()
sr_trial_calc["time_diff"] = sr_trial_calc["relative_time"].diff()
sr_trial_calc["new_trial"] = sr_trial_calc["time_diff"].abs() > .25

mw_trial_calc = mw_trial_calc.copy()
mw_trial_calc["time_diff"] = mw_trial_calc["relative_time"].diff()
mw_trial_calc["new_trial"] = mw_trial_calc["time_diff"].abs() > .25

ctl_trial_calc = ctl_trial_calc.copy()
ctl_trial_calc["time_diff"] = ctl_trial_calc["relative_time"].diff()
ctl_trial_calc["new_trial"] = ctl_trial_calc["time_diff"].abs() > .25


sr_trials = sr_trial_calc["new_trial"].sum() + 1 # add one for first trial
print(sr_trials, " sr trials")

mw_trials = mw_trial_calc["new_trial"].sum() + 1 # add one for first trial
print(mw_trials, " mw trials")

ctl_trials = ctl_trial_calc["new_trial"].sum() + 1 # add one for first trial
print(ctl_trials, " ctl trials")

trial_calc["time_diff"] = trial_calc["relative_time"].diff()
trial_calc["new_trial"] = trial_calc["time_diff"].abs() > .25
num_trials = trial_calc["new_trial"].sum() + 1
print(num_trials, " total trials")
"""

#print(data["label"].unique())



X = data[X_features]

            
y = data["label"] # labels are not binary at this stage
# train test split to get train and holdout sets, keeping label in features for now, will drop after
# setting up for mw onset and self report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = random_state)

X_train = X_train.drop(columns = ["sub_id"])
X_test = X_test.drop(columns = ["sub_id"])

X_train.to_csv(f"X_train_wlen{window_size}.csv")
X_test.to_csv(f"X_test_wlen{window_size}.csv")
y_train.to_csv(f"y_train_wlen{window_size}.csv")
y_test.to_csv(f"y_test_wlen{window_size}.csv")
X.to_csv(f"X_wlen{window_size}.csv")
y.to_csv(f"y_wlen{window_size}.csv")
trial_count_df.to_csv(f"trial_counts_wlen{window_size}.csv")


