#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:42:48 2025

@author: Alaina

1. Use select from Model for initial feature selection 
    * Comparable to using FLDA to find spatial weights in Jangraw 2014
    * Select from model selects features based on importance weights. A threshold 
    is specified and features whose absolute importance is greater or equal are 
    kept while others are discarded: experiment with various thresholds
2. Second level classifier: use the reduced feature set as input to SVM, LDA, 
and Logistic Regression
    * Get AUROC to evaluate
    
    add conf matrices from cv ensemble to plot conf matrices here, logic is same
"""

import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import SelectFromModel


def load_extract_preprocess(filepath, features, random_state = 42):
    """
    
    Loads data, drops missing values, extracts features and labels, and generates
    train and test sets. Assumes that "is_MWreported" is the label column.

    Parameters
    ----------
    filepath : Str. Required.
        The path to the full dataset. This dataset must include the column "is_MWreported".
    features : List of str. Required.
        A list of column names for the features to be used. These columns must be present
        in the file that resides at the filepath destination.
    random_state : Int, optional.
        Random state.

    Returns
    -------
    X_train: Array of float
        Train features.
    X_test: Array of float.
        Test features.
    y_train: Array of bool.
        Train labels.
    y_test: Array of bool.
        Test labels.
    columns : Index.
        Features as ordered in the original dataset.
        
    """
    # load data
    data = pd.read_csv(filepath, index_col=0)
    
    # drop na features for now, INCORPORATE NEW INTERPOLATED DATA ONCE NIHA DONE, should reduce NA
    # see EDA script in this directory.
    clean_data = data.dropna(subset=features)

    # extract features and labels
    # is_MWreported is label
    X = clean_data[features]
    y = clean_data["is_MWreported"].values

    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2,
                                                        random_state = random_state)
    
    columns = X.columns

    return X_train, X_test, y_train, y_test, columns

def load_extract_preprocess_groups(filepath, features, random_state=42):
    """
    
    Loads data, drops missing values, extracts features and labels, and generatees
    train and test sets. Assumes that "is_MWreported" is the label column.
    Additionally, generates and saves a correlation matrix heatmap of features 
    to the current working directory.

    Parameters
    ----------
    filepath : Str. Required.
        The path to the full dataset. This dataset must include the column "is_MWreported".
    features : List of str. Required.
        A list of column names for the features to be used. These columns must be present
        in the file that resides at the filepath destination.
    random_state: Int, optional.
        Random state for ensuring comparable, reproducible results.

    Returns
    -------
    X_train: Array of float
        Train features.
    X_test: Array of float.
        Test features.
    y_train: Array of bool.
        Train labels.
    y_test: Array of bool.
        Test labels.
    groups: Array of int.
        Groups for leave one group out cross validation. Values represent page 
        numbers in the training set.
    columns : Index.
        Features as ordered in the original dataset.
        
    """
    # load data
    data = pd.read_csv(filepath, index_col=0)
    
    # drop na features for now, INCORPORATE NEW INTERPOLATED DATA ONCE NIHA DONE, should reduce NA
    # see EDA script in this directory.
    clean_data = data.dropna(subset=features)

    # extract features and labels
    # is_MWreported is label
    X = clean_data[features]
    y = clean_data["is_MWreported"].values
    
    # add page column for groups
    X = X.copy()
    X["page"] = clean_data["page"]

    
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = random_state)
    
    # groups must be based on train set because test is holdout/ not going through CV
    groups = X_train["page"].values
    
    # drop page from X_train, X and X_test now
    X_train.drop("page", axis=1, inplace=True)
    X_test.drop("page", axis=1, inplace=True)
    X.drop("page", axis=1, inplace=True)
    
    # reset indices
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    
    columns = X.columns
    
    return X_train, X_test, y_train, y_test, groups, columns

def plot_roc(results, threshold, y_test):
    """
    Plots and saves the ROC curve for all models.  Adapted from HS.

    Parameters
    ----------
    results : Dictionary
        Nexted dictionary of results from model training and evaluation.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(12,12))
    bold_leg = []
    index = 0

    thresh_results = results[threshold] # get results for this threshold
    for model_name in thresh_results:
        # handle keys that aren't model names
        if (model_name != "auroc_std") & (model_name != "mean_auroc"):
            # get metrics & extract from lists
            model_y_scores = thresh_results[model_name]["y_scores"]
            auc = thresh_results[model_name]["auroc"]
            fpr, tpr, _ = roc_curve(y_test, model_y_scores)
            
            if auc > 0.9:
                bold_leg.append(index)
            plt.plot(fpr, tpr, label=f'{model_name} AUROC: {auc:.2f}')
            index += 1

    # annotate
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    leg = plt.legend(loc="lower right")
    for index in bold_leg:
        leg.get_texts()[index].set_fontweight('bold')

    plt.title(f'ROC Curves: {threshold} SFM Threshold')
    plt.savefig(f"ROC_curves_SFMGrid_{threshold:.2f}.png")
    plt.show()
    
def evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """
    Trains and evaluates a model, calculating y_scores and auroc

    Parameters
    ----------
    model : SKlearn model. Required.
        The model to be trained and evaluated.
    X_train : Array of float. Required.
        Train features.
    y_train : Array of bool. Required.
        Train labels.
    X_test : Array of float. Required.
        Test features.
    y_test : Array of bool. Required.
        Test labels.

    Returns
    -------
    y_scores : 
        
    auroc : 
    """
    # train
    model.fit(X_train, y_train)

    # get y-scores (samples x classes (as ordered in model.classes_)), 
    # positive class for ROC curve
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:,1]
    else:
        y_scores = model.decision_function(X_test)
    # predict on test
    y_pred = model.predict(X_test)
    
    # get roc-auc
    roc_auc = roc_auc_score(y_test, y_scores)
    
    return y_scores, auroc

def plot_roc_ensemble(results, y_test):
    """
    Plots and saves the ROC curve for all models. 

    Parameters
    ----------
    results : Dictionary
        Nexted dictionary of results from LOPOCV model training and evaluation. 
        First level keys are model names. Second level keys are as follows:
        roc_auc, f1, conf_matrix, true_labels, predicted_probs, tpr, fpr, importances.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(12,12))
    bold_leg = []
    index = 0


    for model_name, metrics in results.items():
        # get metrics & extract from lists
        avg_probs = metrics["avg_probs"]
        fpr, tpr, _ = roc_curve(y_test, avg_probs)
        auc = roc_auc_score(y_test, avg_probs)

        
        if auc > 0.9:
            bold_leg.append(index)
        plt.plot(fpr, tpr, label=f'{model_name} AUROC: {auc:.2f}')
        index += 1

    # annotate
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    leg = plt.legend(loc="lower right")
    for index in bold_leg:
        leg.get_texts()[index].set_fontweight('bold')

    plt.title('ROC Curves: CV Ensemble with SFM Feature Reduction')
    plt.savefig("ROC_curves_SFM_ensemble.png")
    plt.show()
    
def predictor_hist(results, threshold, y_test):
    """
    
    Plot and saves a histogram of the predictor variable for each model. One figure
    is created with a subplot for each model type. Only plotted for one threshold.

    This function calculates the predictor variable: the models predicted probabilities
    for MW classification, and plots a KDE histogram of these predictor variables. The
    data is separated into two distributions based on whether the ground truth label 
    indicates MW or not. Medians of both distributions are displayed, and the model's
    decision boundary is marked. 
    
    Parameters:
    results : Dictionary
        Nexted dictionary of results from LOPOCV model training and evaluation. 
        First level keys are model names. Second level keys are as follows:
        roc_auc, f1, conf_matrix, true_labels, predicted_probs, tpr, fpr, importances.
    ---------
    Returns : 
        None.

    """
    # one figure with a subplot for each model
    # 9 models, arrange in 3x3 grid
    fig, axes = plt.subplots(3,3,figsize=(15,15))
    fig.suptitle(f"Predictor Histograms: {threshold} threshold SFM", fontsize=16)
    # flatten axes
    axes = np.array(axes).flatten()
    # get y_pred for each model
    results = results[threshold] # extract results for the provided threshold
    for idx, (model_name, metrics) in enumerate(results.items()):
        # only do for models
        if (model_name != "auroc_std") & (model_name != "mean_auroc"):

            predictions = metrics["y_scores"]
            labels = y_test
    
            predictions = np.array(predictions)
            labels = np.array(labels)
            # classify preds by labels
            preds_0 = predictions[labels == 0]
            preds_1 = predictions[labels == 1]
            
            ax = axes[idx]
            # plot KDEs
            #sns.kdeplot(preds_0, ax=ax, label="Actually not MW", fill=True, color = "orange")
            #sns.kdeplot(preds_1, ax=ax, label = "Actually MW", fill=True, color = "blue")
            ax.hist(preds_0, label="Actually not MW", color = "orange", alpha=.5, bins=20)
            ax.hist(preds_1, label="Actually MW", color = "blue", alpha=.5, bins=20)
            
            # median markers
            ax.axvline(np.median(preds_0), color = "orange", linestyle = "-", label = "Median (Not MW)")
            ax.axvline(np.median(preds_1), color = "blue", linestyle = "-", label = "Median (MW)")
            
            ax.set_title(f"{model_name}")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Density")
            ax.legend()
        
    plt.tight_layout()      
    plt.savefig(f"Predictor_hist_SFM_{threshold}.png")
    
def predictor_hist_ensemble(results, y_test):
    """
    
    Plot and saves a histogram of the predictor variable for each model. One figure
    is created with a subplot for each model type.
    An analagous figure is created with KDE histograms as well. 

    This function calculates the predictor variable: the models predicted probabilities
    for MW classification, and plots a KDE histogram of these predictor variables. The
    data is separated into two distributions based on whether the ground truth label 
    indicates MW or not. Medians of both distributions are displayed, and the model's
    decision boundary is marked. 
    
    Parameters:
    results : Dictionary
        Nexted dictionary of results from LOPOCV model training and evaluation. 
        First level keys are model names. Second level keys are as follows:
        roc_auc, f1, conf_matrix, true_labels, predicted_probs, tpr, fpr, importances.
    ---------
    Returns : 
        None.

    """
    # one figure with a subplot for each model
    # 9 models, arrange in 3x3 grid
    fig, axes = plt.subplots(3,3,figsize=(15,15))
    fig.suptitle("Predictor Histograms (CV Ensemble with SFM Feature Reduction)", fontsize=16)
    # flatten axes
    axes = np.array(axes).flatten()
    labels = y_test
    # get y_pred for each model
    for idx, (model_name, metrics) in enumerate(results.items()):

        predictions = metrics["avg_probs"]

        predictions = np.array(predictions)
        labels = np.array(labels)
        # classify preds by labels
        preds_0 = predictions[labels == 0]
        preds_1 = predictions[labels == 1]
        
        ax = axes[idx]
        # plot KDEs
        #sns.kdeplot(preds_0, ax=ax, label="Actually not MW", fill=True, color = "orange")
        #sns.kdeplot(preds_1, ax=ax, label = "Actually MW", fill=True, color = "blue")
        ax.hist(preds_0, label="Actually not MW", color = "orange", alpha=.5, bins=20)
        ax.hist(preds_1, label="Actually MW", color = "blue", alpha=.5, bins=20)
        
        # median markers
        ax.axvline(np.median(preds_0), color = "orange", linestyle = "-", label = "Median (Not MW)")
        ax.axvline(np.median(preds_1), color = "blue", linestyle = "-", label = "Median (MW)")
        
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend()
        
    plt.tight_layout()      
    plt.savefig(f"Predictor_hist_SFM_ensemble.png")
    plt.close()
    
    # KDE
    # one figure with a subplot for each model
    # 9 models, arrange in 3x3 grid
    fig, axes = plt.subplots(3,3,figsize=(15,15))
    fig.suptitle("Predictor Histograms (SFM Feature Reduction)", fontsize=16)
    # flatten axes
    axes = np.array(axes).flatten()
    labels = y_test
    # get y_pred for each model
    for idx, (model_name, metrics) in enumerate(results.items()):

        predictions = metrics["avg_probs"]

        predictions = np.array(predictions)
        labels = np.array(labels)
        # classify preds by labels
        preds_0 = predictions[labels == 0]
        preds_1 = predictions[labels == 1]
        
        ax = axes[idx]
        # plot KDEs
        sns.kdeplot(preds_0, ax=ax, label="Actually not MW", fill=True, color = "orange")
        sns.kdeplot(preds_1, ax=ax, label = "Actually MW", fill=True, color = "blue")
        
        # median markers
        ax.axvline(np.median(preds_0), color = "orange", linestyle = "-", label = "Median (Not MW)")
        ax.axvline(np.median(preds_1), color = "blue", linestyle = "-", label = "Median (MW)")
        
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend()
        
    plt.tight_layout()      
    plt.savefig(f"Predictor_hist_SFM_ensemble_KDE.png")
    
def plot_conf_mat(thresh_results, y_test, model_names, thresh, models, X_test):
    """
    Plots confusion matrix heatmaps for up to four models. One figure with a subplot
    for each specified model will be created. The figure is also saved
    to the current working directory.

    Parameters
    ----------
    results : Dictionary
        Dictionary of results from model evaluation.
    y_test: Array of bool.
        True labels
    model_names: List of str.
        The models to plot confusion matrices for.


    Returns
    -------
    None.

    """
    results = thresh_results[thresh]
    # use ensemble results - results from y_scores averaged over cv models
    # one figure with a subplot for each model name in list
    num_models = len(model_names)
    fig, axes = plt.subplots(2,2, figsize=(12,12))
    axes = axes.flatten()
    for subplot_idx, model_name in enumerate(model_names):
        model = models[model_name]
        y_preds = model.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_preds)
        
        #print(conf_mat)
        conf_df = pd.DataFrame(conf_mat / np.sum(conf_mat, axis=1)[:, None], index = ["Not MW", "MW"],
                             columns = ["Not MW", "MW"])

        sns.heatmap(conf_df, annot=True, cmap="coolwarm", vmin = 0, vmax = 1, ax=axes[subplot_idx])
        axes[subplot_idx].set_xlabel("Predicted Value")
        axes[subplot_idx].set_ylabel("True Value")
        axes[subplot_idx].set_title(f"{model_name}")
        
    # hide unused subplots
    for i in range(num_models, len(axes)):  
        axes[i].set_visible(False)
        
    # finalize plot
    fig.suptitle("Confusion Matrices")
    fig.tight_layout()

    plt.savefig("conf_matrices_noCV.png")
    plt.show()
#%%
random_state = 42

features = ["pupil_slope", "norm_pupil", "norm_fix_word_num", "norm_sac_num", 
            "norm_in_word_reg", "norm_out_word_reg", "zscored_zipf_fixdur_corr",
            "norm_total_viewing", "zscored_word_length_fixdur_corr", "blink_num",
            "blink_dur"]

filepath = "group_R_features_same-dur.csv"

# Load & Split Data
X_train, X_test, y_train, y_test, columns = load_extract_preprocess(filepath,
                                                                    features,
                                                                    random_state)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SFM
    # train RF, get feat. importance
rf = RandomForestClassifier(random_state = random_state) # consider adjusting n_estimators if want to optimize
rf.fit(X_train, y_train)
importances = rf.feature_importances_
    # feature importance threshold values to test using with SFM
    # RF feature importances without CV shows no features are below .04 anyways
    # so range from .03 to max importance, testing 10 values (we only have 11 features total)
thresholds = np.linspace(0.03, max(importances), num=10)
    # define level 2 classifiers
models = {
        'Logistic Regression': LogisticRegression(class_weight="balanced", random_state = random_state),
        'Support Vector Machine': SVC(kernel="linear", probability=True, class_weight="balanced", random_state = random_state),
        'Decision Tree': DecisionTreeClassifier(class_weight="balanced", random_state = random_state),
        'Random Forest': RandomForestClassifier(class_weight="balanced", random_state = random_state),
        'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier()
    }
    # set up result storage structure for results of each model at each threshold
    # ultimately will be nested dictionary with a top level key for each threshold
    # second level keys are model names with values as AUROC for that model with SFM
    # at that threshold
threshold_results = {}

    # train with the different thresholded importanes
for thresh in thresholds:
    selector = SelectFromModel(rf, threshold=thresh, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    # set up result storage structure for all models for this threshold - keys are model names
    model_results = {}
    # loop through models, training and evaluating each
    for model_name, model in models.items():
        model.fit(X_train_selected, y_train) # train
        y_scores = model.predict_proba(X_test_selected)[:,1]# evaluate on test set, get probs for pos. class
        auroc = roc_auc_score(y_test, y_scores) # get auroc
        model_results[model_name] = { # store results
            "auroc" : auroc,
            "y_scores" : y_scores.tolist()}
    # add results from all models with this SFM threshold to results dict
    threshold_results[thresh] = model_results

# find best threshold - find mean AUROC as well as std. Best threshold
# is that with highest mean auroc and low std (consistently good performance 
# across models)
mean_aurocs = []
auroc_std = []
for thresh in thresholds:
    this_thresh_aurocs = []
    for model_name in models:
        # gather auroc for each model at this threshold, appending to list
        model_auroc = threshold_results[thresh][model_name]["auroc"]
        this_thresh_aurocs.append(model_auroc)
    # mean auroc for each threshold
    mean_auroc = np.mean(this_thresh_aurocs)
    # auroc std for each threshold 
    auroc_std = np.std(this_thresh_aurocs)
    # add mean and std auroc to this threshold key
    threshold_results[thresh]["mean_auroc"] = mean_auroc
    threshold_results[thresh]["auroc_std"] = auroc_std

# highest mean auroc threshold - find max mean auroc, print corresponding key
# do same for lowest std
max_mean_auroc = 0 # start at 0
max_mean_auroc_thresh = None
max_mean_auroc_std = 0
min_std_auroc = float("inf") # very high to start
min_std_auroc_thresh = None
max_auroc = 0
max_auroc_thresh = None
max_auroc_model = None
for thresh in thresholds:
    mean_auroc = threshold_results[thresh]["mean_auroc"]
    auroc_std = threshold_results[thresh]["auroc_std"]
    # save over max if this is higher, save thresh 
    if mean_auroc > max_mean_auroc:
        max_mean_auroc = mean_auroc
        max_mean_auroc_thresh = thresh
        max_mean_auroc_std = auroc_std
    # save over min if this is lower, save thresh
    if auroc_std < min_std_auroc:
        min_std_auroc = auroc_std
        min_std_auroc_thresh = thresh
    # loop through each model for this threshold, looking at raw auroc
    for model_name in models:
        auroc = threshold_results[thresh][model_name]["auroc"]
        if auroc > max_auroc:
            max_auroc = auroc
            max_auroc_thresh = thresh
            max_auroc_model = model_name

if max_mean_auroc_thresh is not None:
    print(f"The SFM threshold associated with the highest mean auroc across model types was: {max_mean_auroc_thresh}. The mean AUROC was: {max_mean_auroc}")
    print(f"The auroc std associated with this threshold was {max_mean_auroc_std}")
else:
    print("Something happened when trying to find the threshold associated with the max mean auroc. Figure it out.")
"""
# lowest std auroc threshold
if min_std_auroc_thresh is not None:
    print(f"The SFM threshold associated with the lowest auroc std across model types was: {min_std_auroc_thresh}. The AUROC std was: {min_std_auroc}")
else:
    print("Something happened when trying to find the threshold associated with the min auroc std. Figure it out.")
"""
# threshold with max auroc and corresponding model
if max_auroc_thresh is not None:
    print(f"The SFM threshold associated with highest auroc across model types was: {max_auroc_thresh}. The AUROC was: {max_auroc}. The model type was {max_auroc_model}")
else:
    print("Something happened when trying to find the threshold associated with the max auroc. Figure it out.")


# plot roc curve with auroc for each threshold
for thresh in thresholds:
    plot_roc(threshold_results, thresh, y_test)


# proceed with threshold associated with highest mean auroc, raise a flag if 
# threshold associated with highest mean auroc != threshold associated with highest
# auroc, but still proceed

if max_mean_auroc_thresh != max_auroc_thresh:
    print("The SFM threshold associated with the highest mean auroc is not the same as the threshold associated with the max auroc. Take a closer look.")
    
predictor_hist(threshold_results, max_auroc_thresh, y_test)
model_names = ["Logistic Regression", "Support Vector Machine", "Linear Discriminant Analysis"]
#%%
plot_conf_mat(threshold_results, y_test, model_names, max_auroc_thresh, models, X_test_selected)
#%% 
"""
Doesn't really make sense to do CV now because feature importance differs over folds
so the selector chooses different features even if the same threshold is used
and we should have the selector per-fold to avoid data leakage, right? The problem
is that due to the differences in data in each fold, we get different numbers
of features retained. RFE could support this since its based on number of
features retained rather than a threshold.
"""
# reload data, new split with groups
X_train, X_test, y_train, y_test, groups, columns = load_extract_preprocess_groups(filepath, features, random_state)

# Scale holdout set
scaler = StandardScaler()
scaler.fit(X_train) # NO TRANSFORMING TRAIN YET, DO IN CV LOOP
X_test = scaler.transform(X_test) # transform holdout set

# reduce holdout set
rf = RandomForestClassifier(random_state = random_state) # consider adjusting n_estimators if want to optimize
rf.fit(X_train, y_train)
importances = rf.feature_importances_

selector = SelectFromModel(rf, threshold=max_mean_auroc_thresh, prefit=True) # USES MAX MEAN AUROC THRESH HERE
X_test_selected = selector.transform(X_test)

# show which features are still present with this threshold?

# now, train cv ensemble using feature set reduced with this threshold
logo = LeaveOneGroupOut()
cv_results = {model_name:{
    "y_scores" : [],
    "auroc" : []}for model_name in models.keys()}

# test results will store cv model results on test set (to be averaged for ensemble predictions)
test_results = {model_name:{
    "y_scores" : []} for model_name in models.keys()}

for train_idx, test_idx in logo.split(X_train,y_train,groups):
    # train test split for CV
    X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]
    
    # scale features for this fold
    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_test_fold = scaler.transform(X_test_fold)
    
    # reduce features for this fold
    rf = RandomForestClassifier(random_state = random_state)
    rf.fit(X_train_fold, y_train_fold)
    importances = rf.feature_importances_

    selector = SelectFromModel(rf, threshold=max_mean_auroc_thresh, prefit=True) # USES MAX MEAN AUROC THRESH HERE
    X_train_fold_selected = selector.transform(X_train_fold)
    X_test_fold_selected = selector.transform(X_test_fold)
    
    for model_name, model in models.items():
        # evaluate is called for each fold, so metrics generated for each fold
        y_scores, auroc = evaluate(model, X_train_fold_selected, y_train_fold, X_test_fold_selected, y_test_fold, model_name)
        
        cv_results[model_name]["y_scores"].append(y_scores)
        cv_results[model_name]["auroc"].append(auroc)

        # get y_scores on holdout set for ensemble predictions
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test_selected)[:,1]
        else:
            y_scores = model.decision_function(X_test_selected)
        test_results[model_name]["y_scores"].append(y_scores)
        
# ensemble results - holds y_probs averaged over cv model folds
ensemble_results = {model_name:{
    "avg_probs" : None
    }for model_name in models.keys()}
        
# do the ensembling - weighted average depending on foldwise auroc
for model_name, model in models.items():
    # get auc for each fold's model's test set evaluation
    fold_aurocs = [] # 1 auroc for each fold
    all_folds_y_scores = test_results[model_name]["y_scores"]
    for fold_scores in all_folds_y_scores:
        fold_auc = roc_auc_score(y_test, fold_scores)
        fold_aurocs.append(fold_auc)
        
    # get info for weighted arithmetic mean of aurocs and intercepts/ weights
    # normalization not needed due to weighted arithmetic mean
    averaging_weights = fold_aurocs #np.array(fold_aurocs) / np.sum(fold_aurocs)
    weighting_denominator = np.sum(averaging_weights)
    
    # weighted arithmetic mean of y_scores over folds dependent on auc
    all_fold_probs_arr = np.array(test_results[model_name]["y_scores"]) # rows are folds, cols are samples
    weighted_scores_numerator = np.zeros_like(all_fold_probs_arr) 

    # loop through each fold for this model type
    for fold_idx, fold_scores in enumerate(test_results[model_name]["y_scores"]):
        # mult. the auc for this fold by the scores for this fold's model on test set
        weighted_scores_numerator += averaging_weights[fold_idx] * fold_scores
        
    weighted_scores = weighted_scores_numerator / weighting_denominator
    ensemble_results[model_name]["avg_probs"] = weighted_scores[0,:] # grab the first row bc they get repeated
    
# plot roc curves with auroc
plot_roc_ensemble(ensemble_results, y_test)

# predictor histograms
predictor_hist(ensemble_results, y_test)
    
