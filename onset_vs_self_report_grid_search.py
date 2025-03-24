#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:20:06 2025

@author: Alaina

Performs grid search to find optimal hyperparameters for logistic regression,
support vector machine, linear discriminant analysis, and random forest when used
in the onset_vs_self_report_ pipeline.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.exceptions import ConvergenceWarning
import warnings

#%%       

def run_grid_search(X_train, y_train, model, param_grid, cv=10, scoring="roc_auc", 
                    random_state=42):
    # preprocessing: scale, smote, pca
    steps = [
        # scale
        ("scaler", StandardScaler()),
        # smote 
        ("smote", "passthrough"),
        # pca
        ("pca", "passthrough"),
        # classifier
        ("classifier", model)
        ]
    
    # initialize pipeline
    pipeline = Pipeline(steps)
    
    # run grid search
    grid = GridSearchCV(pipeline, param_grid = param_grid, scoring = scoring, cv = cv, return_train_score = True) 
    # tradeoff with the cost of additional computational expense
    

    grid.fit(X_train, y_train)

    
    return grid
        

        

#%%
    

window_size = 5 # options are two and five. mainly focusing on 5 at this point
random_state = 42 # shouldn't need to change this
    
# not including smote as hp because convergence usually fails without it

# define hyperparams to test for each model
# these will be tested for each classifier type
# all models: PCA .8 variance retained, PCA .9 variance retained,
# PCA .95 variance retained, no PCA.
# Logistic Regression: penalty, C, solver
# SVM: C, no gamma because linear
# LDA: solver
param_grids = {
    "Logistic Regression" : {
        "smote": [SMOTE(random_state=random_state), "passthrough"],
        "pca": [PCA(n_components=.8, random_state = random_state),
                PCA(n_components = .9, random_state = random_state),
                PCA(n_components = .95, random_state = random_state),
                "passthrough"],
        "classifier__penalty": ["l2"], #only l2 supported by all solvers. can try more solvers depending on which penalty is optimal
        "classifier__C": [.0001, .001, .01, .1, 1, 10, 100],
        "classifier__solver": ["lbfgs", "liblinear"], 
        },
    "Support Vector Machine": {
        "smote": [SMOTE(random_state=random_state), "passthrough"],
        "pca": [PCA(n_components=.8, random_state = random_state),
                PCA(n_components = .9, random_state = random_state),
                PCA(n_components = .95, random_state = random_state),
                "passthrough"],
        "classifier__C": [.0001, .001, .01, .1, 1, 10, 100]
        },
    "Linear Discriminant Analysis": {
        "smote": [SMOTE(random_state=random_state), "passthrough"],
        "pca": [PCA(n_components=.8, random_state = random_state),
                PCA(n_components = .9, random_state = random_state),
                PCA(n_components = .95, random_state = random_state),
                "passthrough"],
        "classifier__solver": ["svd", "lsqr"]
        }
    }
"""
    "Random Forest": {
        "classifier__n_estimators": [10, 100, 200],
        "classifier__max_depth": [None, 5, 10, 20]
        }}
,
    "AdaBoost": {
        "classifier__n_estimators": [10, 50, 100, 200],  
        "classifier__learning_rate": [0.001, 0.01, 0.1, 1], 
        "classifier__algorithm": ["SAMME", "SAMME.R"]  
        },
    "Naive Bayes": {
        "classifier__var_smoothing": [.000000001, .00000001, .0000001, .000001, .00001]
        },
    "XGBoost": {
        "classifier__n_estimators": [50, 100, 200], 
        "classifier__learning_rate": [0.001, 0.01, 0.1, 0.3], 
        "classifier__max_depth": [3, 5, 10], 
        "classifier__subsample": [0.5, 0.7, 1.0],  
        "classifier__colsample_bytree": [0.5, 0.7, 1.0]  
        }
    }
"""

# load data
X_train = pd.read_csv(f"X_train_wlen{window_size}", index_col=0)
y_train = pd.read_csv(f"y_train_wlen{window_size}", index_col=0).squeeze("columns")

# train test split to get train and holdout sets, keeping label in features for now, will drop after
# setting up for mw onset and self report
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = random_state)


# from train set, filter out the following - leave the test set untouched
# rows where label = control and relative_time != 0
# rows where label = self report and relative_time != -1 if window_size = 2, -2.5 if window_size = 5
# rows where label = mw_onset and relative_time != 0

X_train = X_train[
    ((X_train["label"] == "control") & (X_train["relative_time"] == 0)) |
    ((X_train["label"] == "self_report") & (X_train["relative_time"] == (-.5 * window_size))) |
    ((X_train["label"] == "MW_onset") & (X_train["relative_time"] == 0)) |
    ((X_train["label"] == "MW_onset") & (X_train["relative_time"] == 2)) | # add condition for mw2_target
    ((X_train["label"] == "control") & (X_train["relative_time"] == 2)) |
    ((X_train["label"] == "control") & (X_train["relative_time"] == (-.5 * window_size)) |
    ((X_train["label"] == "MW_onset") & (X_train["relative_time"] == 5)) |
    ((X_train["label"] == "control") & (X_train["relative_time"] == 5)))
    ] # retain some control times at rel time mw2 target and -.5* window as well



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

# do the same for the filtered x test sets

X_train_MW_onset = X_train[(X_train["label"] != "self_report") & (X_train["relative_time"] == 0)]
X_train_MW_onset = X_train_MW_onset.copy()
X_train_MW_onset = X_train_MW_onset.drop(columns=["label", "page", "relative_time"])


# create X_train_MW_onset_2
X_train_MW_onset_2 = X_train[(X_train["label"] != "self_report") & (X_train["relative_time"] == 2)]
X_train_MW_onset_2 = X_train_MW_onset_2.copy()
X_train_MW_onset_2 = X_train_MW_onset_2.drop(columns=["label", "page", "relative_time"])

# create X_train_MW_onset_5
X_train_MW_onset_5 = X_train[(X_train["label"] != "self_report") & (X_train["relative_time"] == 5)]
X_train_MW_onset_5 = X_train_MW_onset_5.copy()
X_train_MW_onset_5 = X_train_MW_onset_5.drop(columns=["label", "page", "relative_time"])

# create X_train_self_report: drop rows where label = mw onset (retain self report and control), then drop label & page
X_train_self_report = X_train[(X_train["label"] != "MW_onset") & (X_train["relative_time"] == (-.5 * window_size))]
X_train_self_report = X_train_self_report.copy()
X_train_self_report = X_train_self_report.drop(columns=["label", "page", "relative_time"]) 


# drop relative time from X_train and X_test now
X_train_relative_times = X_train["relative_time"].copy()
X_train = X_train.drop(columns=["relative_time"])

# drop pages and labels from X_train. X_train isn't used again aside from fitting the scaler for the holdout set
# but we need to do this to make that work. relative time has already been dropped
X_train = X_train.drop(columns=["label", "page"])


#correlation_matrix = X_train.corr(method='pearson')
#correlation_matrix.to_csv("corr_matrix.csv") 

# verify idx match for X_train_relative_times and y_train so rel times can be used
# to filter y_train for mw onset
if X_train_relative_times.index.equals(y_train.index):
    print("Indexes for X train relative times and y train match!")
else:
    print("Indexes do not match (X train relative times and y train).")
    

    
# at this stage there are only mw events for rel time 2 in y train


# create y_train_MW_onset: 0 for control, 1 for MW_onset when rel time = 0. Rows where label = self_report dropped
y_train_MW_onset = y_train[X_train_relative_times == 0]
y_train_MW_onset = y_train_MW_onset[y_train_MW_onset != "self_report"]
y_train_MW_onset = y_train_MW_onset.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_MW_onset_2: : 0 for control, 1 for MW_onset when rel time = 2. Rows where label = self_report dropped
y_train_MW_onset_2 = y_train[X_train_relative_times == 2]
y_train_MW_onset_2 = y_train_MW_onset_2[y_train_MW_onset_2 != "self_report"]
y_train_MW_onset_2 = y_train_MW_onset_2.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_MW_onset_2: : 0 for control, 1 for MW_onset when rel time = 2. Rows where label = self_report dropped
y_train_MW_onset_5 = y_train[X_train_relative_times == 5]
y_train_MW_onset_5 = y_train_MW_onset_5[y_train_MW_onset_5 != "self_report"]
y_train_MW_onset_5 = y_train_MW_onset_5.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_self_report: 0 for control, 1 for self_report. Rows where label = MW_onset dropped
y_train_self_report = y_train[X_train_relative_times == (-.5 * window_size)]
y_train_self_report = y_train_self_report[y_train_self_report != "MW_onset"]
y_train_self_report = y_train_self_report.apply(lambda x: 1 if x in ["self_report"] else 0)


# define models

self_report_models = {
        'Logistic Regression': LogisticRegression( random_state = random_state, max_iter = 1000),
        'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state),
        #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
        #'Random Forest': RandomForestClassifier(random_state = random_state),
        #'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        #'KNN': KNeighborsClassifier(),
        #'Naive Bayes': GaussianNB(),
        #'XGBoost': XGBClassifier(random_state = random_state)
    }

MW_onset_models = {
        'Logistic Regression': LogisticRegression(random_state = random_state, max_iter=1000),
        'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state),
        #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
        #'Random Forest': RandomForestClassifier(random_state = random_state),
        #'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        #'KNN': KNeighborsClassifier(),
        #'Naive Bayes': GaussianNB(),
        #'XGBoost': XGBClassifier(random_state = random_state)
    }

MW_onset_2_models = {
        'Logistic Regression': LogisticRegression(random_state = random_state, max_iter = 1000),
        'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state),
        #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
        #'Random Forest': RandomForestClassifier(random_state = random_state),
        #'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        #'KNN': KNeighborsClassifier(),
        #'Naive Bayes': GaussianNB(),
        #'XGBoost': XGBClassifier(random_state = random_state)
    }
MW_onset_5_models = {
        'Logistic Regression': LogisticRegression(random_state = random_state),
        'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state),
        #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
        #'Random Forest': RandomForestClassifier(random_state = random_state),
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
        },
    "MW_onset_5": {
        "X": X_train_MW_onset_5,
        "y": y_train_MW_onset_5,
        "models": MW_onset_5_models
        }
    }

# initialize dict for grid search result storage
# first level keys- classifier types
# second level keys - model names
# third level keys - AUROC (best validation set auroc found) & Params (associated hyperparameters)
best_results = {}
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
                                   cv=10, scoring="roc_auc", random_state=random_state)
            
            # initialize inner dict for this model key
            best_results[classifier_type][model_name] = {}
            cv_results = pd.DataFrame(grid.cv_results_)
            
            best_idx = cv_results["mean_test_score"].idxmax() # get index of best mean auroc over cv folds
            best_mean_auroc = cv_results.loc[best_idx, "mean_test_score"] # get the best mean auroc over cv folds
            best_params = cv_results.loc[best_idx, "params"] # then get the corresponding hyperparameters
            
            best_results[classifier_type][model_name]["mean_AUROC"] = best_mean_auroc
            best_results[classifier_type][model_name]["Params"] = best_params
            
            #best_results[classifier_type][model_name]["AUROC"] = grid.best_score_
            #best_results[classifier_type][model_name]["Params"] = grid.best_params_
            print(f"Best Parameters for {model_name} ({classifier_type})")
            print(best_params)
            print(f"Best Mean AUROC Over CV Folds: {best_mean_auroc:.4f}")
        
results_df = pd.DataFrame(best_results)
results_df.to_csv(f"./onset_v_self_report/grid_search_results_{window_size}s_window.csv")
    