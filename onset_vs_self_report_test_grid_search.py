#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:20:06 2025

@author: Alaina

Performs grid search to find optimal hyperparameters for logistic regression,
support vector machine, linear discriminant analysis, and random forest when used
in the onset_vs_self_report_test pipeline.

Grid search is performed with PCA retaining 90% of variance and SMOTE with 
auto sampling strategy because those choices led to improved performance
in models prior to hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from matplotlib import pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline

#%%          
def run_grid_search(X_train, y_train, model, param_grid, cv=3, scoring="roc_auc", 
                    random_state=42, PCA_flag=True, PCA_threshold=.9, SMOTE_flag=True):
    # preprocessing: scale, smote, pca
    steps = []
    steps.append(("scaler", StandardScaler()))
    
    if SMOTE_flag:
        steps.append(("smote", SMOTE(random_state=random_state)))
    else:
        steps.append(("smote", "passthrough")) # pass if smote is off
    
    if PCA_flag:
        steps.append(("pca", PCA(n_components = PCA_threshold, random_state=random_state)))
    else:
        steps.append(("pca", "passthrough")) # pass if pca is off
        
    # model time
    steps.append(("classifier", model))
    
    # initialize pipeline
    pipeline = Pipeline(steps)
    
    # run grid search
    grid = GridSearchCV(pipeline, param_grid = param_grid, scoring = scoring, cv = cv, return_train_score = True) 
    # tradeoff with the cost of additional computational expense
    grid.fit(X_train, y_train)
    
    return grid
        

#%%
    

window_size = 5 # options are two and five. 2 has less imbalance
random_state = 42 # shouldn't need to change this
SMOTE_flag = True # set to True to use SMOTE (synthetic minority over-sampling technique) to achieve more balanced class dist.
undersample_flag = False # set to True to use undersampling to achieve more balanced class dist. 
PCA_flag = True
PCA_threshold = .9 # for PCA
# DO NOT SET BOTH UNDERSAMPLE_FLAG AND SMOTE_FLAG TO TRUE AT THE SAME TIME THAT WOULD BE WEIRD/ ERROR WILL BE THROWN

assert not (SMOTE_flag == True and undersample_flag == True), "Error: cannot use SMOTE and undersampling at the same time this way. Change one to False or adapt the pipeline."
    
# define hyperparams to test for each model
# these will be tested for each classifier type
# Logistic Regression: penalty, C, solver
# SVM: C, no gamma because linear
# LDA: solver
# RF: num trees, depth
param_grids = {
    "Logistic Regression" : {
        "classifier__penalty": ["l2"], #only l2 supported by all solvers. can try more solvers depending on which penalty is optimal
        "classifier__C": [.001, .01, .1, 1, 10],
        "classifier__solver": ["lbfgs", "liblinear", "sag", "saga"]
        },
    "Support Vector Machine": {
        "classifier__C": [.001, .01, .1, 1, 10]
        },
    "Linear Discriminant Analysis": {
        "classifier__solver": ["svd", "lsqr", "eigen"]
        },
    "Random Forest": {
        "classifier__n_estimators": [10, 100, 200],
        "classifier__max_depth": [None, 5, 10, 20]}
    }

# load data
filepath = f"group_R_features_slide_wlen{window_size}.csv"
data = pd.read_csv(filepath,index_col=0)


#print(data.isna().sum())



# handle missing values - drop horizontal_sacc then drop the nas
data.drop(columns="horizontal_sacc", inplace=True)


# feature extraction - create X and y for splitting
# labels is included in features at this stage, but dropped later (same with page)


features = ["fix_num","label", "norm_fix_word_num", "norm_in_word_reg",
            "norm_out_word_reg", "zscored_zipf_fixdur_corr", "zipf_fixdur_corr",
            "zscored_word_length_fixdur_corr","norm_total_viewing", "fix_dispersion",
            "weighted_vergence","blink_num", "blink_dur", "blink_freq", "norm_sacc_num",
            "sacc_length","norm_pupil", "page", "relative_time"]

data.dropna(subset = features, inplace=True)

#print(data["label"].unique())



X = data[features]
y = data["label"] # labels are not binary at this stage
# train test split to get train and holdout sets, keeping label in features for now, will drop after
# setting up for mw onset and self report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = random_state)


# from train set, filter out the following - leave the test set untouched
# rows where label = control and relative_time != 0
# rows where label = self report and relative_time != -1 if window_size = 2, -2.5 if window_size = 5
# rows where label = mw_onset and relative_time != 0

X_train = X_train[
    ((X_train["label"] == "control") & (X_train["relative_time"] == 0)) |
    ((X_train["label"] == "self_report") & (X_train["relative_time"] == (-.5 * window_size))) |
    ((X_train["label"] == "MW_onset") & (X_train["relative_time"] == 0)) |
    ((X_train["label"] == "MW_onset") & (X_train["relative_time"] == 2)) | # add condition for mw onset + 2
    ((X_train["label"] == "control") & (X_train["relative_time"] == 2)) |
    ((X_train["label"] == "control") & (X_train["relative_time"] == (-.5 * window_size)))] # retain some control times at rel time 2 and -.5* window as well


# sanity checks.. should have all three label types and only 0 as relative time 
# val for each group except self_report, which should have only -1 or -2.5 depending on window size.
print("unique vals for label column:", X_train["label"].unique())
print("unique vals for relative time, groupedby label: ")
print(X_train.groupby("label")["relative_time"].unique())

# adjust y_train accordingly so that it matches full X_train (same rows dropped)
y_train = y_train.loc[X_train.index]


# drop rel time from each train set after creating

# create X_train_MW_onset: only keep rows where label isn't self report and relative time is 0
# then drop label & page. Also create groups for CV

X_train_MW_onset = X_train[(X_train["label"] != "self_report") & (X_train["relative_time"] == 0)]
X_train_MW_onset = X_train_MW_onset.copy()
#MW_onset_groups = X_train_MW_onset["page"].values
X_train_MW_onset.drop(columns=["label", "page", "relative_time"], inplace=True)

# create X_train_MW_onset_2
X_train_MW_onset_2 = X_train[(X_train["label"] != "self_report") & (X_train["relative_time"] == 2)]
X_train_MW_onset_2 = X_train_MW_onset_2.copy()
#MW_onset_2_groups = X_train_MW_onset_2["page"].values
X_train_MW_onset_2.drop(columns=["label", "page", "relative_time"], inplace=True)

# create X_train_self_report: drop rows where label = mw onset (retain self report and control), then drop label & page
X_train_self_report = X_train[(X_train["label"] != "MW_onset") & (X_train["relative_time"] == (-.5 * window_size))]
X_train_self_report = X_train_self_report.copy()
#self_report_groups = X_train_self_report["page"].values
X_train_self_report.drop(columns=["label", "page", "relative_time"], inplace=True) 

# drop relative time from X_train and X_test now
X_train_relative_times = X_train["relative_time"].copy()
X_train.drop(columns=["relative_time"], inplace=True)
X_test_relative_times = X_test["relative_time"].copy()
#X_test_relative_times.reset_index(inplace=True, drop=True)
X_test.drop(columns=["relative_time"], inplace=True)




# drop labels from X_test
X_test.drop(columns=["label", "page"], inplace=True)

# drop pages and labels from X_train. X_train isn't used again aside from fitting the scaler for the holdout set
# but we need to do this to make that work. relative time has already been dropped
X_train.drop(columns=["label", "page"], inplace=True)

#correlation_matrix = X_train.corr(method='pearson')
#correlation_matrix.to_csv("corr_matrix.csv") 

# verify idx match for X_train_relative_times and y_train so rel times can be used
# to filter y_train for mw onset
if X_train_relative_times.index.equals(y_train.index):
    print("Indexes match!")
else:
    print("Indexes do not match.")
    
# at this stage there are only mw events for rel time 2 in y train


# create y_train_MW_onset: 0 for control, 1 for MW_onset when rel time = 0. Rows where label = self_report dropped
y_train_MW_onset = y_train[X_train_relative_times == 0]
y_train_MW_onset = y_train_MW_onset[y_train_MW_onset != "self_report"]
y_train_MW_onset = y_train_MW_onset.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_MW_onset_2: : 0 for control, 1 for MW_onset when rel time = 2. Rows where label = self_report dropped
y_train_MW_onset_2 = y_train[X_train_relative_times == 2]
y_train_MW_onset_2 = y_train_MW_onset_2[y_train_MW_onset_2 != "self_report"]
y_train_MW_onset_2 = y_train_MW_onset_2.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_self_report: 0 for control, 1 for self_report. Rows where label = MW_onset dropped
y_train_self_report = y_train[X_train_relative_times == (-.5 * window_size)]
y_train_self_report = y_train_self_report[y_train_self_report != "MW_onset"]
y_train_self_report = y_train_self_report.apply(lambda x: 1 if x in ["self_report"] else 0)

# create a label set where all original labels are retained for plotting later
# this is not to be used for any training or testing, just as a source of truth for plotting
y_test_copy = y_test.copy()
true_labels_test_set = y_test_copy.apply(lambda x: 1 if x == "MW_onset" else (2 if x == "self_report" else 0))
# binarize y_test: 0 for control, 1 for self_report OR mw_onset
y_test = y_test.apply(lambda x: 1 if x in ["MW_onset", "self_report"] else 0)

# define models

self_report_models = {
        'Logistic Regression': LogisticRegression( random_state = random_state),
        'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state),
        #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
        'Random Forest': RandomForestClassifier(random_state = random_state),
        #'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        #'KNN': KNeighborsClassifier(),
        #'Naive Bayes': GaussianNB(),
        #'XGBoost': XGBClassifier(random_state = random_state)
    }

MW_onset_models = {
        'Logistic Regression': LogisticRegression(random_state = random_state),
        'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state),
        #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
        'Random Forest': RandomForestClassifier(random_state = random_state),
        #'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        #'KNN': KNeighborsClassifier(),
        #'Naive Bayes': GaussianNB(),
        #'XGBoost': XGBClassifier(random_state = random_state)
    }

MW_onset_2_models = {
        'Logistic Regression': LogisticRegression(random_state = random_state),
        'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state),
        #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
        'Random Forest': RandomForestClassifier(random_state = random_state),
        #'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        #'KNN': KNeighborsClassifier(),
        #'Naive Bayes': GaussianNB(),
        #'XGBoost': XGBClassifier(random_state = random_state)
    }

# map classifier types to models and train data 
# first level key classifier type
# second level keys X (train features), y (train labels), models (model dict for that classifier type)
data_model_dict = {
    "self_report": {
        "X": X_train_self_report,
        "y": y_train_self_report,
        "models": self_report_models
        },
    "MW_onset": {
        "X": X_train_MW_onset,
        "y": y_train_MW_onset,
        "models": MW_onset_models
        },
    "MW_onset_2": {
        "X": X_train_MW_onset_2,
        "y": y_train_MW_onset_2,
        "models": MW_onset_2_models
        }
    }

# initialize dict for grid search result storage
# first level keys- classifier types
# second level keys - model names
# third level keys - AUROC (best validation set auroc found) & Params (associated hyperparameters)
best_results = {}

# grid search: loop over classifier types
for classifier_type, data in data_model_dict.items():
    print(f"Performing Grid Search for {classifier_type} models...")
    # grab x train and y train
    X_train_curr = data["X"]
    y_train_curr = data["y"]
    # initialize inner dict for this classifier key
    best_results[classifier_type] = {}
    # grid search: loop over models within classifier type
    for model_name, model in data["models"].items():
        print(f"Performing Grid Search for {model_name} ({classifier_type})")
        grid = run_grid_search(X_train_curr, y_train_curr, model, param_grids[model_name],
                               cv=3, scoring="roc_auc", random_state=random_state,
                               PCA_flag=PCA_flag, PCA_threshold=.9, SMOTE_flag=SMOTE_flag)
        # initialize inner dict for this model key
        best_results[classifier_type][model_name] = {}
        best_results[classifier_type][model_name]["AUROC"] = grid.best_score_
        best_results[classifier_type][model_name]["Params"] = grid.best_params_
        print(f"Best Parameters for {model_name} ({classifier_type})")
        print(grid.best_params_)
        print(f"Best AUROC: {grid.best_score_:.4f}")
    
results_df = pd.DataFrame(best_results)
results_df.to_csv("./onset_v_self_report/OnTest/grid_search_results.csv")
    
    