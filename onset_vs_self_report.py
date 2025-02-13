#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:20:06 2025

@author: Alaina

Evaluates the self-report task paradigm as described in 
https://glassbrainlab.wordpress.com/2025/02/10/mr-analysis-methods/

Two separate classifiers are trained with LOPOCV (leave one page out cross validation),
one for self_report vs. control and one for MW_onset vs. control. Samples labelled 
as self_report are discluded from the train set for the MW_onset classifiers and 
vise-versa. Then, both classifiers are evaluated on a test set including all 
samples regardless of their labels.

A plot of probabilities for the positive class is then created. The 
plot will have two lines, one for probabilities coming from the self_report
classifier and the other for probabilities coming from the mw_onset classifier.
If the lines are similar, it suggests that MW_onset and self_report share
similar features. For each line, probabilities are aligned with their time relative to 
the midpoint of the window & aggregated over overlapping windows (for aggregation,
the mean is taken. Standard Error is also represented through shading). 


All rows where relative time !=0 should be dropped so that events are only considered
to occur in window time around actual event. This is true for all except the 
self_report labels, these are the end of the page so there is no case where
relative_time = 0 here. Instead, self_report relative time should be 1/2 window size.
This is only done for the train set, all realtive times are left in the test/ holdout
set for more interesting analysis.


Summary:
    - 9 model types are trained in two different ways with LOGOCV.
        - 1. self_report models: trained on a dataset with self_report events
        as the positive class and mw_onset events omitted.
        - 2. mw_onset models: trained on a dataset with mw_onset events
        as the positive class and self_report events omitted.
    - The "best model" for both classifier types (self report vs control and mw onset vs control)
    is assessed through looking at average LOGOCV performance in terms of AUROC and F1 score.
    - All 9 model types for both classifier types are then evaluated on a test set
    where both mw_onset and self_report are considered to be the positive class.
        - Resulting predicted probabilities of the positive class are plotted in a set
        of three subplots. Each subplot shows predicted probabilities aggregated
        over relative time such that the mean is taken over predicted probabilities 
        that correspond to the same relative time. The control subplot shows this
        for when the event was control, the self_report subplot shows this for when the
        event was self_report, and the MW_onset subplot shows this for when the event was MW_onset.
        
Issue?
How to really analyze behavior around event onset for self_report since we don't 
have samples after that?
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

def logo_cv(X_train, y_train, groups, models):
    """
    Perform leave-one-group-out cross validation (LOGOCV) with pages as groups 
    on the provided models.

    Parameters
    ----------
    X_train : DataFrame
        Features for training.
    y_train : Series
        Labels for training.
    groups : Array of int.
        Groups for LOGOCV.
    models : Dictionary
        Models to train. Keys are model names.

    Returns
    -------
    models : Dictionary
        Trained models. Keys are model names.
    cv_results : Dictionary
        Nested dictionary of foldwise results from cross validation. First level
        keys are model names as specified in the models dictionary. Second level 
        keys are lists of size equal to the number of folds, named "y_scores_fold"
        and "true_labels_fold" which hold the predicted probabilities of the positive
        class and the true labels respectively. The values at each list index
        correspond to the fold number for that list index.

    """

    # set up result storage structure 
    cv_results = {model_name:{
        "y_scores_fold" : [],
        "true_labels_fold" : [] }for model_name in models.keys()}
    
    logo = LeaveOneGroupOut()
    
    for train_idx, test_idx in logo.split(X_train,y_train,groups):

        X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

        # scale features for this fold
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        for model_name, model in models.items():
            model.fit(X_train_fold, y_train_fold)
            # get y-scores
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test_fold)[:,1]
            else:
                y_scores = model.decision_function(X_test_fold)
                
            # save the info
            cv_results[model_name]["y_scores_fold"].append(y_scores)
            cv_results[model_name]["true_labels_fold"].append(y_test_fold)
            
    return models, cv_results

def get_AUROC(y_test, self_report_models, mw_onset_models, test_results):
    """
    Compute area under the roc curve (AUROC) on the test set for all self_report 
    vs. control models as well as all MW_Onset vs. control models.

    Parameters
    ----------
    y_test : Series
        True labels for the test set.
    self_report_models : Dictionary
        Models trained to classify self_report vs. control. First level keys are 
        model names.
    mw_onset_models : Dictionary
        Models trained to classify mw_onset vs. control. First level keys are
        model names.
    test_results : Dictionary
        Nested dictionary of results from evaluation on the test set. First level
        keys are model names, matching those seen in self_report_models and
        mw_onset_models. Second level keys are MW_onset_y_scores and self_report_y_scores.
        Values are predicted probabilities for the positive class when each model
        was evaluated on the test set. MW_onset_y_scores holds y_scores for the model
        when trained to distinguish between mw_onset vs. control while self_report_y_scores
        holds y_scores for the model when trained to distinguish between self_report vs. control.

    Returns
    -------
    aurocs : Dictionary
        Nested dictionary holding the area under the ROC curve for each model. 
        First level keys are model names, matching those seen in test_results.
        Second level keys are MW_onset_auroc and self_report_auroc, which hold 
        scalars (float).

    """
    
    # set up storage structure
    aurocs = {model_name:{
        "self_report_auroc" : None,
        "MW_onset_auroc" : None} for model_name in self_report_models.keys()}
    
    # get auroc for all self report models
    for model_name, model in self_report_models.items():
        y_scores = test_results[model_name]["self_report_y_scores"]
        roc_auc = roc_auc_score(y_test, y_scores)
        aurocs[model_name]["self_report_auroc"] = roc_auc
        
    # get auroc for all mw onset models
    for model_name, model in mw_onset_models.items():
        y_scores = test_results[model_name]["MW_onset_y_scores"]
        roc_auc = roc_auc_score(y_test, y_scores)
        aurocs[model_name]["MW_onset_auroc"] = roc_auc
    
    return aurocs
    
def get_f1(y_test, X_test, self_report_models, mw_onset_models):
    """
    Compute the F1-score on the test set for all self_report vs. control models 
    as well as all MW_Onset vs. control models.

    Parameters
    ----------
    y_test : Series
        True labels for the test set.
    X_test : Array of float.
        Features for the test set.
    self_report_models : Dictionary
        Models trained to classify self_report vs. control. First level keys are 
        model names.
    mw_onset_models : Dictionary
        Models trained to classify mw_onset vs. control. First level keys are
        model names.

    Returns
    -------
    f1s : Dictionary
        Nested dictionary holding the F1-scores for each model. 
        First level keys are model names, matching those seen in test_results.
        Second level keys are MW_onset_f1 and self_report_f1, which hold 
        scalars (float).

    """
    
    # set up storage structure
    f1s = {model_name:{
        "self_report_f1" : None,
        "MW_onset_f1" : None} for model_name in self_report_models.keys()}
    
    # get f1 for all self report models
    for model_name, model in self_report_models.items():
        y_preds = model.predict(X_test)
        f1 = f1_score(y_test, y_preds)
        f1s[model_name]["self_report_f1"] = f1
    
    # get f1 for all mw onset models
    for model_name, model in mw_onset_models.items():
        y_preds = model.predict(X_test)
        f1 = f1_score(y_test, y_preds)
        f1s[model_name]["MW_onset_f1"] = f1
        
    return f1s

def plot_auroc_f1(aurocs, f1s, window_size):
    """
    Generate a grouped bar chart of AUROC and F1-Score for each model type, with
    subplots for the classifier type (MW_onset vs. control or self_report vs. control)
    
    Parameters
    ----------
    aurocs : Dictionary
        Nested dictionary holding the area under the ROC curve for each model. 
        First level keys are model names, matching those seen in test_results.
        Second level keys are MW_onset_auroc and self_report_auroc, which hold 
        scalars (float).
    f1s : Dictionary
        Nested dictionary holding the F1-scores for each model. 
        First level keys are model names, matching those seen in test_results.
        Second level keys are MW_onset_f1 and self_report_f1, which hold 
        scalars (float).
    window_size : int
        The sliding window size used throughout the pipeline.

    Returns
    -------
    None.
    """

    
    # model names are first level keys of either dictionary
    models = aurocs.keys()
    # initialize lists
    MW_onset_aurocs = []
    self_report_aurocs = []
    MW_onset_f1s = []
    self_report_f1s = []
    
    for model_name in models:
        # get aurocs
        MW_onset_aurocs.append(aurocs[model_name]["MW_onset_auroc"])
        self_report_aurocs.append(aurocs[model_name]["self_report_auroc"])
        # get f1s
        MW_onset_f1s.append(f1s[model_name]["MW_onset_f1"])
        self_report_f1s.append(f1s[model_name]["self_report_f1"])
        
    # get maxs for bolding
    max_mw_onset_auroc = max(MW_onset_aurocs)
    max_mw_onset_f1 = max(MW_onset_f1s)
    max_self_report_auroc = max(self_report_aurocs)
    max_self_report_f1 = max(self_report_f1s)
    
    # handle colors based on max
    auroc_color = "lightsteelblue"
    f1_color = "burlywood"
                             
    x = np.arange(len(models))
    # side by side plots for MW_onset and self_report models
    fig, axes = plt.subplots(1,2, figsize = (17,9), sharey=True, sharex=True)
    fig.suptitle(f"All Models AUROC and F1: {window_size}s Sliding Window")
    mwo_auroc_bars = axes[0].bar(x-.2, MW_onset_aurocs, .4, label="AUROC", color = auroc_color)
    mwo_f1_bars = axes[0].bar(x+.2, MW_onset_f1s, .4, label="F1", color = f1_color)
    axes[0].set_title("MW_Onset Models")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha="right")
    axes[0].set_ylabel("Score")
    #axes[0].legend(loc=8)
    
    sr_auroc_bars = axes[1].bar(x-.2, self_report_aurocs, .4, label="AUROC", color = auroc_color)
    sr_f1_bars = axes[1].bar(x+.2, self_report_f1s, .4, label="F1", color = f1_color)
    axes[1].set_title("self_report Models")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha="right")
    axes[1].set_ylabel("Score")
    #axes[1].legend(loc=8)
    
    add_labels(mwo_auroc_bars, MW_onset_aurocs, max_mw_onset_auroc, axes[0])
    add_labels(mwo_f1_bars, MW_onset_f1s, max_mw_onset_f1, axes[0])
    add_labels(sr_auroc_bars, self_report_aurocs, max_self_report_auroc, axes[1])
    add_labels(sr_f1_bars, self_report_f1s, max_self_report_f1, axes[1])
    
    # remove top border so labels are all visible
    # as well as right side border so it looks nice
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
    # get legend for overall figure since its the same for both subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    
    plt.tight_layout(rect=[0,0,.95,1])
    plt.savefig(f"./onset_v_self_report/{window_size}s/auroc_f1_onset_self_report_{window_size}s_win")
    plt.show()
    
def add_labels(bars, values, max_val, ax):
    """
    Add value labels to bars in a bar chart.

    Parameters
    ----------
    bars : Matplotlib bar container
        The bars for the bar chart.
    values : List
        The values/ heights for the bars in the bar chart.
    max_val : Float
        The value to highlight in the labels.
    ax : Matplotlib axes object
        The axes to annotate.

    Returns
    -------
    None.

    """
    for bar in bars:
        height = bar.get_height()
        
        # bold max vals
        if height == max_val:
            font_weight = "bold"
        else:
            font_weight = "normal"
            
        ax.text(bar.get_x() + bar.get_width()/2, height + .02, f"{height:.2f}",
                ha='center', va='bottom', fontsize=10, fontweight = font_weight)
        
        
def plot_probs(test_results, X_test_relative_times, true_labels_test_set, window_size):
    """
    Plot predicted probabilities for the positive class as a function of 
    relative time. For each model type, a figure will be created with three 
    subplots: one to show predicted probabilities of control events, one to
    show predicted probabilitieswhen the true event was MW_onset, and one to show predicted
    probabilities when the true event was self_report. For each subplot, there are two lines.
    One line shows aggregated (mean over matching relative_times) predicted 
    probabilities of the positive class from models trained to distinguish 
    between MW_onset and control, while the other shows aggregated (mean over 
    matching relative_times) predicted probabilities of the positive class
    from models trained to distinguish between self_report and control. For each 
    subplot, the standard error over the aggregated probabilities is also shown.
    
    Parameters
    ----------
    test_results : Dictionary
        Nested dictionary of results from evaluation on the test set. First level
        keys are model names, matching those seen in self_report_models and
        mw_onset_models. Second level keys are MW_onset_y_scores and self_report_y_scores.
        Values are predicted probabilities for the positive class when each model
        was evaluated on the test set. MW_onset_y_scores holds y_scores for the model
        when trained to distinguish between mw_onset vs. control while self_report_y_scores
        holds y_scores for the model when trained to distinguish between self_report vs. control.
    X_test_relative_times : Series
        Relative times of samples in X_test, with indices matching X_test.
    true_labels_test_set : Series
        True labels for the test set. This should not be the same as the labels used for 
        evaluation that contain a 1 for MW_Onset OR self_report. Instead, this should
        have 1 for MW_onset, 2 for self_report, and 0 for control.
    window_size : Int.
        The size of the sliding window used throughout the pipeline.

    Returns
    -------
    None.

    """

    print("Before modifying relative times in function:")
    print(X_test_relative_times[true_labels_test_set == 2].unique())

    # adjust relative times to handle self_report offset (since the event center
    # for those is - 1/2 window size)
    X_test_relative_times_adjusted = X_test_relative_times.copy()  # dont alter OG
    X_test_relative_times_adjusted.loc[true_labels_test_set == 2] += (0.5 * window_size)
    
    print("After modifying relative times in function:")
    print(X_test_relative_times_adjusted[true_labels_test_set == 2].unique())


    # merge relative times and y test on index
    metadata = pd.concat([X_test_relative_times_adjusted, true_labels_test_set], axis=1)
    plot_types = ["Control", "MW_Onset", "self_report"]
    
    
    # create these subplots for each model type
    for model_name, results in test_results.items():
        fig, axes = plt.subplots(2,2, figsize =(15,15), sharey=True, sharex=True)
        fig.suptitle(f"{model_name}: Predicted Probabilities as a Function of Relative Time, {window_size}s Sliding Window")
        
        axes = axes.flatten()
        fig.delaxes(axes[3]) # get rid of the empty subplot
        # get predicted probs for both classifier types, retaining indices
        # since these are all probabilities for each classifier type from X_test, 
        # indices should match those in the full series of relative times
        MW_onset_probs = pd.Series(results["MW_onset_y_scores"], index = X_test_relative_times_adjusted.index)
        self_report_probs = pd.Series(results["self_report_y_scores"], index = X_test_relative_times_adjusted.index)

        # make subplot for each type
        for subplot_idx, plot_type in enumerate(plot_types):
            ax = axes[subplot_idx]
            # filter data based on plot type
            if plot_type == "MW_Onset":
                subset = metadata[metadata["label"] == 1]
            elif plot_type == "self_report":
                subset = metadata[metadata["label"] == 2]
            else: # control
                subset = metadata[metadata["label"] == 0]
                
            # sort by relative time so x-axis makes sense
            subset = subset.sort_values("relative_time")

            # get mean over each relative time and SE for each classifier type
            MW_onset_mean_probs = subset.groupby("relative_time").apply(lambda x: np.mean(MW_onset_probs[x.index]))
            MW_onset_probs_se = subset.groupby("relative_time").apply(lambda x: MW_onset_probs[x.index].sem())
            
            self_report_mean_probs = subset.groupby("relative_time").apply(lambda x: np.mean(self_report_probs[x.index]))
            self_report_probs_se = subset.groupby("relative_time").apply(lambda x: self_report_probs[x.index].sem())
            # plot lines and SE- index of mean probs is relative times bc of groupby
            # MW onset
            ax.plot(MW_onset_mean_probs.index, MW_onset_mean_probs, label="MW_onset Classifier")
            ax.fill_between(MW_onset_mean_probs.index,
                            MW_onset_mean_probs - MW_onset_probs_se,
                            MW_onset_mean_probs + MW_onset_probs_se,
                            alpha = .4)
            # self report
            ax.plot(self_report_mean_probs.index, self_report_mean_probs, label="self_report Classifier")
            ax.fill_between(self_report_mean_probs.index,
                            self_report_mean_probs - self_report_probs_se,
                            self_report_mean_probs + self_report_probs_se,
                            alpha = .4)
            
            # subplot details
            ax.set_title(f"{plot_type}")
            ax.set_xlabel("Window Midpoint Relative to Event Onset (s)")
            if subplot_idx %2 == 0:
                ax.set_ylabel("Predicted Probability of Positive Class")
            # dashed line for event onset
            ax.axvline(0, color="black", linestyle = "--", label="Event Onset")
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(f"./onset_v_self_report/{window_size}s/{model_name}_probabilities_{window_size}s_win")
        plt.show()
    
    
#%%
    

window_size = 5 # options are two and five
random_state = 42

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
            "sacc_length","mean_pupil","norm_pupil", "page", "relative_time"]

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
    ((X_train["label"] == "self_report") & (X_train["relative_time"] == (-.5*window_size))) |
    ((X_train["label"] == "MW_onset") & (X_train["relative_time"] == 0))]


# sanity checks.. should have all three label types and only 0 as relative time 
# val for each group except self_report, which should have only -1 or -2.5 depending on window size.
print("unique vals for label column:", X_train["label"].unique())
print("unique vals for relative time, groupedby label: ")
print(X_train.groupby("label")["relative_time"].unique())

# adjust y_train accordingly
y_train = y_train.loc[X_train.index]

# drop relative time from X_train and X_test now
X_train.drop(columns=["relative_time"], inplace=True)
X_test_relative_times = X_test["relative_time"].copy()
#X_test_relative_times.reset_index(inplace=True, drop=True)
X_test.drop(columns=["relative_time"], inplace=True)

# create X_train_MW_onset: drop rows where label = self_report, then drop label & page
X_train_MW_onset = X_train[X_train["label"] != "self_report"]
X_train_MW_onset = X_train_MW_onset.copy()
MW_onset_groups = X_train_MW_onset["page"].values
X_train_MW_onset.drop(columns=["label", "page"], inplace=True)


# create X_train_self_report: drop rows where label = self_report, then drop label & page
X_train_self_report = X_train[X_train["label"] != "MW_onset"]
X_train_self_report = X_train_self_report.copy()
self_report_groups = X_train_self_report["page"].values
X_train_self_report.drop(columns=["label", "page"], inplace=True)

# drop labels from X_test
X_test.drop(columns=["label", "page"], inplace=True)

# drop pages and labels from X_train. X_train isn't used again aside from fitting the scaler for the holdout set
# but we need to do this to make that work.
X_train.drop(columns=["label", "page"], inplace=True)

correlation_matrix = X_train.corr(method='pearson')
correlation_matrix.to_csv("corr_matrix.csv")

# scale holdout set - train sets will be scaled in cv loop to avoid data leakage
scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)

# create y_train_MW_onset: 0 for control, 1 for MW_onset. Rows where label = self_report dropped
y_train_MW_onset = y_train[y_train != "self_report"]
y_train_MW_onset = y_train_MW_onset.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_self_report: 0 for control, 1 for self_report. Rows where label = MW_onset dropped
y_train_self_report = y_train[y_train != "MW_onset"]
y_train_self_report = y_train_self_report.apply(lambda x: 1 if x in ["self_report"] else 0)

# create a label set where all original labels are retained for plotting later
# this is not to be used for any training or testing, just as a source of truth for plotting
y_test_copy = y_test.copy()
true_labels_test_set = y_test_copy.apply(lambda x: 1 if x == "MW_onset" else (2 if x == "self_report" else 0))
# binarize y_test: 0 for control, 1 for self_report OR mw_onset
y_test = y_test.apply(lambda x: 1 if x in ["MW_onset", "self_report"] else 0)

# define models
self_report_models = {
        'Logistic Regression': LogisticRegression(class_weight="balanced", random_state = random_state),
        'Support Vector Machine': SVC(kernel="linear", probability=True, class_weight="balanced", random_state = random_state),
        'Decision Tree': DecisionTreeClassifier(class_weight="balanced", random_state = random_state),
        'Random Forest': RandomForestClassifier(class_weight="balanced", random_state = random_state),
        'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(random_state = random_state)
    }

MW_onset_models = {
        'Logistic Regression': LogisticRegression(class_weight="balanced", random_state = random_state),
        'Support Vector Machine': SVC(kernel="linear", probability=True, class_weight="balanced", random_state = random_state),
        'Decision Tree': DecisionTreeClassifier(class_weight="balanced", random_state = random_state),
        'Random Forest': RandomForestClassifier(class_weight="balanced", random_state = random_state),
        'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(random_state = random_state)
    }


# sanity checks
print(f"MW_Onset Features: {X_train_MW_onset.columns}")
print(f"Self Report Features: {X_train_self_report.columns}")

# train self report vs. control
# reset indices to ensure alignment
X_train_self_report.reset_index(inplace=True, drop=True)
y_train_self_report = y_train_self_report.reset_index(drop=True)
self_report_models, cv_self_report_results = logo_cv(X_train_self_report,
                                                     y_train_self_report,
                                                     self_report_groups,
                                                     self_report_models)
        
# train MW_onset vs. control - logocv
X_train_MW_onset.reset_index(inplace=True, drop=True)
y_train_MW_onset = y_train_MW_onset.reset_index(drop=True)
MW_onset_models, cv_MW_onset_results = logo_cv(X_train_MW_onset,
                                                     y_train_MW_onset,
                                                     MW_onset_groups,
                                                     MW_onset_models)

# evaluate self_report classifier on test/ holdout set

# set up result storage structure for both classifier types: nested dict
# with first level keys as model names, second level keys self_report and mw_onset y_scores
test_results = {model_name:{
    "self_report_y_scores" : None,
    "MW_onset_y_scores" : None} for model_name in self_report_models.keys()}


for model_name, model in self_report_models.items():
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:,1]
    else:
        y_scores = model.decision_function(X_test)
        
    # add to results
    test_results[model_name]["self_report_y_scores"] = y_scores

# evaluate mw_onset classifier on test/ holdout set
for model_name, model in MW_onset_models.items():
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:,1]
    else:
        y_scores = model.decision_function(X_test)
        
    # add to results
    test_results[model_name]["MW_onset_y_scores"] = y_scores

# get AUROC and F1 for test set. Bring this up with group- is this the best way or should we assess
# best model based on CV results? Keep in mind different feature sets are used for training and testing
aurocs = get_AUROC(y_test, self_report_models, MW_onset_models, test_results)
f1s = get_f1(y_test, X_test, self_report_models, MW_onset_models)

# display auroc and f1 in grouped bar chart 
plot_auroc_f1(aurocs, f1s, window_size)
# prob look to rf for now, think we should narrow down features eventually

# plot probabilities from self_report on holdout set and mw_onset on holdout set
# feed this the true_labels_test_set so it has different indicators for self_report and mw_onset
plot_probs(test_results, X_test_relative_times, true_labels_test_set, window_size)

