#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:20:06 2025

@author: Alaina

Evaluates the self-report task paradigm as described in 
https://glassbrainlab.wordpress.com/2025/02/10/mr-analysis-methods/

This script finds raw scores from the full dataset whereas onset_vs_self_report_test
finds raw scores from the test set.


Three separate classifiers are trained after an 80/20 train/test split. The classifier
types are as follows:
    - self_report: train data is filtered so that only rows where relative time = -.5 * window_size
    and the label is not mw_onset are retained. This results in all train samples having
    a relative time of -.5 * window_size and a label of either control or self report.
    - mw_onset: train data is filtered so that only rows where relative time = 0
    and the label is not self_report are retained. This results in all train samples
    having a relative time of 0 and a label of either control or mw_onset.
    - mw_onset_2: train data is filtered so that only rows where relative time = 2
    and the label is not self_report are reatined. This results in all train samples
    having a relative time of 2 and a label of either control or mw_onset. The 
    intention here is to train the classifier on samples where features
    of MW would be stronger- we theorize that this would be after MW has "settled"
    for a moment rather than immediately on self reported onset. 
    
For the test set, all samples are retained, regardless of their event or relative time.

Plots of raw scores are then created where raw scores are the dot product of 
linear model weights and full dataset features. Each plot has three lines, one for raw
scores from the self_report classifier, one for raw scores from the mw_onset 
classifier, and one for raw scores from the mw_onset_2 classifier.
If the lines are similar, it suggests that MW_onset and self_report share
similar features. Trends in these plots can also provide insight into the MW 
paradigm as a whole. For each line, scores are aligned with their time relative to 
the midpoint of the window & aggregated over overlapping windows (for aggregation,
the mean is taken. Standard Error is also represented through shading). 

Two types of performance metrics are calculated. Unfiltered AUROC and F1 represent
AUROC and F1 score computed from the test set resulting from an 80/20 split, but 
this test set isnt filtered by time or label like the train sets were. This means 
y-scores that are ultimately used to calculate AUROC are the result of models 
trained on certain relative times trying to make classifications on a set that includes 
all relative times. This means that the unfiltered AUROC and F1 reflects how well the models 
generalize to all times. Filtered AUROC and F1 represent AUROC and F1 score
computed from the test set that was filtered to match the labels and times seen in the 
train sets after an 80/20 split. This means y-scores used to calculate AUROC
are the result of models trained on certain relative times trying to make classifications
on a set that only includes those relative times. This means that the filtered AUROC
and F1 reflect how well the models perform at specifically identifying self report
events at relative time -.5*window_size, mw_onset events at relative time 0, and mw_onset events
at relative time 2 respectively.

3/6 Note: Finding optimal test metrics when retaining 90% of variance with PCA
and using SMOTE with auto sampling strategy.
3/10 Note: Optimal hyperparameters have been found through grid search and 
used here. Search was performed with SMOTE and PCA retaining 90% of variance.
        
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
# SMOTE implementation reference: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/


def train_models(X_train, y_train, X_test, X_test_filtered, models, PCA_flag, random_state, threshold, full_feature_set):
    """
    Train the provided models without cross validation. Reduces the train and test
    sets with PCA according to the provided threshold if specified.

    Parameters
    ----------
    X_train : DataFrame
        Features for training.
    y_train : Series
        Labels for training.
    X_test : DataFrame
        Features for testing.
    models : Dictionary
        Models to train. Keys are model names.
    PCA_flag : Boolean
        Boolean value speciying whether or not to perform principal component
        analysis for dimensionality reduction. Set to True to turn PCA on, False
        to turn PCA off.
    random_state: Int.
        The random state used throughout the pipeline.
    threshold: Float.
        The threshold for percent variance to retain after PCA. For example, 
        specify .9 to retain 90% of the variance in the data after dimensionality 
        reduction.
    full_feature_set: DataFrame
        Full set of features prior to train test split. Columns must match
        training data. This data will only be changed if PCA is on because
        the same transformation must be applied here as was applied to the 
        train set.
    X_test_filtered: DataFrame
        Features for testing with rows filtered to match the X_train set specific to
        this classifier type.

    Returns
    -------
    models : Dictionary
        Trained models. Keys are model names.
    results : Dictionary
        Nested dictionary of test results. First level keys are model names 
        as specified in the models dictionary. Second level keys are "y_scores"
        and hold the y_scores for on the full, unfiltered test set for the model denoted by the
        first level key.
    X_test : DataFrame
        Features for testing. This is only changed through this function if
        PCA_flag is set to true, because the same transformation applied to the
        train set must be applied to the test set.
    full_feature_set: DataFrame
        Features from the train and test set. THis is only chaned through this
        funciton if PCA_flag is set to true, because the same transformation 
        applied to the train set must be applied tot he test set. 
    filtered_results: Dictionary
        Nested dictionary of test results. First level keys are model names 
        as specified in the models dictionary. Second level keys are "y_scores"
        and hold the y_scores for on the filtered test set for the model denoted by the
        first level key.
    X_test_filtered: DataFrame
        Features for testing, filtered to match X_train. This is only changed 
        through this function if PCA_flag is set to true, because the same 
        transformation applied to the train set must be applied to the test set.  

    """
    # set up result storage structure 
    results = {model_name:{
        "y_scores" : None}for model_name in models.keys()}
    
    filtered_results = {model_name:{
        "y_scores" : None}for model_name in models.keys()}
    
    # scale features 
    scaler = StandardScaler()
    #print("X_train cols: ", X_train.columns)
    #print("X_test cols: ", X_test.columns)
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns = X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    X_test_filtered = pd.DataFrame(scaler.transform(X_test_filtered), index = X_test_filtered.index, columns = X_test_filtered.columns)
    full_feature_set = pd.DataFrame(scaler.transform(full_feature_set), index = full_feature_set.index, columns = full_feature_set.columns)
    
    PCA_threshold = threshold # threshold for % variance to retain after PCA
    
    if PCA_flag:
        """
        # dimensionality reduction/ feature selection
        # find top k components for this fold (explain ~x% of variance)
        # train PCA on train set, transform test set
        pca = PCA(random_state = random_state)
        pca.fit(X_train) # just fitting, not transforming at this point because want to investigate how many components to keep
        fold_variance_ratio = pca.explained_variance_ratio_
        #print(variance_ratio)
        threshold = PCA_threshold
        cumsum = 0
        rat_idx = 0 # index for variance ratio list
        while cumsum < threshold:
            cumsum += fold_variance_ratio[rat_idx]
            rat_idx += 1
        
        print(f"{rat_idx} components explain {cumsum * 100: .2f}% of variance")
        """
        
        # reduce data, saving indices
        
        # save indices
        X_train_indices = X_train.index
        X_test_indices = X_test.index
        X_test_filtered_indices = X_test_filtered.index
        full_feature_set_indices = full_feature_set.index

        #num_components = rat_idx 
        reduce_pca = PCA(n_components = PCA_threshold, random_state = random_state)
        # fit transform train
        X_train = pd.DataFrame(reduce_pca.fit_transform(X_train),index = X_train_indices)
        # transform test
        X_test = pd.DataFrame(reduce_pca.transform(X_test), index = X_test_indices)
        
        X_test_filtered = pd.DataFrame(reduce_pca.transform(X_test_filtered), index = X_test_filtered_indices)
        # transform full set
        full_feature_set = pd.DataFrame(reduce_pca.transform(full_feature_set), index = full_feature_set_indices)
        
    
    # compute y scores with test set scaled the same way as current train set 
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        # get y-scores
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:,1]
            y_scores_filtered = model.predict_proba(X_test_filtered)[:,1]
        else:
            y_scores = model.decision_function(X_test)
            y_scores_filtered = model.decision_function(X_test_filtered)
            
        # save the info
        results[model_name]["y_scores"] = y_scores
        filtered_results[model_name]["y_scores"] = y_scores_filtered
    
            
    return models, results, X_test, full_feature_set, filtered_results, X_test_filtered

def get_AUROC(y_test, self_report_models, mw_onset_models, mw_onset_2_models, test_results):
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
        Models trained to classify mw_onset vs. control with events at relative
        time 0. First level keys are model names.
    mw_onset_2_models : Dictionary
        Models trained to classify mw_onset vs. control with events at relative 
        time 2. First level keys are model names.
    test_results : Dictionary
        Nested dictionary of results from evaluation on the test set. First level
        keys are model names, matching those seen in self_report_models and
        mw_onset_models. Second level keys are MW_onset_y_scores, 
        MW_onset_2_y_scores, and self_report_y_scores. Values are predicted 
        probabilities for the positive class when each model was evaluated on 
        the test set. For example, MW_onset_y_scores holds y_scores for the model
        when trained to distinguish between mw_onset vs. control while self_report_y_scores
        holds y_scores for the model when trained to distinguish between self_report vs. control.

    Returns
    -------
    aurocs : Dictionary
        Nested dictionary holding the area under the ROC curve for each model. 
        First level keys are model names, matching those seen in test_results.
        Second level keys are MW_onset_auroc, MW_onset_2_auroc, and self_report_auroc, 
        which hold scalars (float).

    """
    
    # set up storage structure
    aurocs = {model_name:{
        "self_report_auroc" : None,
        "MW_onset_auroc" : None,
        "MW_onset_2_auroc" : None} for model_name in self_report_models.keys()}
    
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
        
    # get auroc for all mw onset 2 models
    for model_name, model in mw_onset_2_models.items():
        y_scores = test_results[model_name]["MW_onset_2_y_scores"]
        roc_auc = roc_auc_score(y_test, y_scores)
        aurocs[model_name]["MW_onset_2_auroc"] = roc_auc
    
    return aurocs

def get_one_AUROC(y_test, models, test_results, classifier_type):
    """
    Compute area under the roc curve (AUROC) on the test set for all models for
    one classifier type (self_report, MW_onset, or MW_onset_2). If computing
    AUROC on filtered test set, this function must be used because the test sets
    are different for each classifier type.
    
    y_test : Series
        True labels for the test set.
    models : Dictionary
        Models trained to classify one event type vs. control. First level keys are 
        model names.
    test_results : Dictionary
        Nested dictionary of results from evaluation on the test set. First level
        keys are model names, matching those seen in self_report_models and
        mw_onset_models. Second level keys are MW_onset_y_scores, 
        MW_onset_2_y_scores, and self_report_y_scores. Values are predicted 
        probabilities for the positive class when each model was evaluated on 
        the test set. For example, MW_onset_y_scores holds y_scores for the model
        when trained to distinguish between mw_onset vs. control while self_report_y_scores
        holds y_scores for the model when trained to distinguish between self_report vs. control.
    classifier_type: String
        The type of classifier that auroc is being computed for. self_report, MW_onset, 
        or MW_onset_2

    Returns
    -------
    aurocs : Dictionary
        Dictionary holding the area under the ROC curve for each model. 
        First level keys are model names, matching those seen in test_results.
        Values are auroc for each model (float).
    """
    # initialize dict
    aurocs = {model_name: None for model_name in self_report_models.keys()}
    
    # get auroc
    for model_name, model in models.items():
        y_scores = test_results[model_name][f"{classifier_type}_y_scores"]
        roc_auc = roc_auc_score(y_test, y_scores)
        aurocs[model_name] = roc_auc
        
    return aurocs

    
def get_f1(y_test, X_test_sr, X_test_mw, X_test_mw2, self_report_models, mw_onset_models, mw_onset_2_models):
    """
    Compute the F1-score on the test set for all models.

    Parameters
    ----------
    y_test : Series
        True labels for the test set.
    X_test_sr : DataFrame.
        Features for the test set. If PCA_flag was True for training, this test
        set has been transformed using the same PCA model as the train set for 
        self_report. Otherwise it is identical to X_test_mw and X_test_mw2.
    X_test_mw : DataFrame.
        Features for the test set. If PCA_flag was True for training, this test
        set has been transformed using the same PCA model as the train set for 
        mw_onset. Otherwise it is identical to X_test_sr and X_test_mw2.
    X_test_mw2 : DataFrame.
        Features for the test set. If PCA_flag was True for training, this test
        set has been transformed using the same PCA model as the train set for 
        mw_onset_2. Otherwise it is identical to X_test_mw and X_test_sr.
    self_report_models : Dictionary
        Models trained to classify self_report vs. control. First level keys are 
        model names.
    mw_onset_models : Dictionary
        Models trained to classify mw_onset vs. control. First level keys are
        model names.
    mw_onset_2_models : Dictionary
        Models trained to classify mw_onset_2 vs. control. First level keys are
        model names.

    Returns
    -------
    f1s : Dictionary
        Nested dictionary holding the F1-scores for each model. 
        First level keys are model names, matching those seen in test_results.
        Second level keys are MW_onset_f1, MW_onset_2_f1, and self_report_f1, 
        which hold scalars (float).

    """
    
    # set up storage structure
    f1s = {model_name:{
        "self_report_f1" : None,
        "MW_onset_f1" : None,
        "MW_onset_2_f1" : None} for model_name in self_report_models.keys()}
    
    # get f1 for all self report models
    for model_name, model in self_report_models.items():
        y_preds = model.predict(X_test_sr)
        f1 = f1_score(y_test, y_preds)
        f1s[model_name]["self_report_f1"] = f1
    
    # get f1 for all mw onset models
    for model_name, model in mw_onset_models.items():
        y_preds = model.predict(X_test_mw)
        f1 = f1_score(y_test, y_preds)
        f1s[model_name]["MW_onset_f1"] = f1
        
    # get f1 for all mw onset 2 models
    for model_name, model in mw_onset_2_models.items():
        y_preds = model.predict(X_test_mw2)
        f1 = f1_score(y_test, y_preds)
        f1s[model_name]["MW_onset_2_f1"] = f1
        
    return f1s

def get_one_f1(y_test, X_test, models):
    """
    Compute the F1-score on the test set for all models, for one classifier type.
    (self_report, MW_onset, or MW_onset_2). If computing F1 on filtered test set, 
    this function must be used because the test sets are different for each classifier type.
    
    Parameters
    ----------
    y_test : Series
        True labels for the test set.
    X_test : DataFrame.
        Features for the test set. I
    models : Dictionary
        Models trained to classify one event type vs. control. First level keys are
        model names.

    Returns
    -------
    f1s : Dictionary
        Dictionary holding the F1-scores for each model. 
        First level keys are model names, matching those seen in models. F1 scores 
        are values.

    """
    
    # set up storage structure
    f1s = {model_name: None for model_name in self_report_models.keys()}
    
    # get f1 for all self report models
    for model_name, model in models.items():
        y_preds = model.predict(X_test)
        f1 = f1_score(y_test, y_preds)
        f1s[model_name] = f1
        
    return f1s


def plot_auroc_f1(aurocs, f1s, window_size, filtered):
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
    filtered: Bool
        A flag to designate whether the displayed auroc and f1 were computed 
        from teh filtered test set or from the unfiltered test set.

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
    MW_onset_2_aurocs = []
    MW_onset_2_f1s = []
    
    for model_name in models:
        # get aurocs
        MW_onset_aurocs.append(aurocs[model_name]["MW_onset_auroc"])
        MW_onset_2_aurocs.append(aurocs[model_name]["MW_onset_2_auroc"])
        self_report_aurocs.append(aurocs[model_name]["self_report_auroc"])
        # get f1s
        MW_onset_f1s.append(f1s[model_name]["MW_onset_f1"])
        MW_onset_2_f1s.append(f1s[model_name]["MW_onset_2_f1"])
        self_report_f1s.append(f1s[model_name]["self_report_f1"])
        
        
    # get maxs for bolding
    max_mw_onset_auroc = max(MW_onset_aurocs)
    max_mw_onset_f1 = max(MW_onset_f1s)
    max_self_report_auroc = max(self_report_aurocs)
    max_self_report_f1 = max(self_report_f1s)
    max_mw_onset_2_auroc = max(MW_onset_2_aurocs)
    max_mw_onset_2_f1 = max(MW_onset_2_f1s)
    
    # handle colors based on max
    auroc_color = "lightsteelblue"
    f1_color = "burlywood"
                             
    x = np.arange(len(models))
    # side by side plots for MW_onset and self_report models
    fig, axes = plt.subplots(1,3, figsize = (17,9), sharey=True, sharex=True)
    if filtered:
        fig.suptitle(f"All Models Filtered Test Set AUROC and F1: {window_size}s Sliding Window")
    else:
        fig.suptitle(f"All Models Unfiltered Test Set AUROC and F1: {window_size}s Sliding Window")
        
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
    
    mwo_2_auroc_bars = axes[2].bar(x-.2, MW_onset_2_aurocs, .4, label="AUROC", color = auroc_color)
    mwo_2_f1_bars = axes[2].bar(x+.2, MW_onset_2_f1s, .4, label="F1", color = f1_color)
    axes[2].set_title("MW_Onset 2 Models")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha="right")
    axes[2].set_ylabel("Score")
    
    add_labels(mwo_auroc_bars, MW_onset_aurocs, max_mw_onset_auroc, axes[0])
    add_labels(mwo_f1_bars, MW_onset_f1s, max_mw_onset_f1, axes[0])
    add_labels(sr_auroc_bars, self_report_aurocs, max_self_report_auroc, axes[1])
    add_labels(sr_f1_bars, self_report_f1s, max_self_report_f1, axes[1])
    add_labels(mwo_2_auroc_bars, MW_onset_aurocs, max_mw_onset_2_auroc, axes[2])
    add_labels(mwo_2_f1_bars, MW_onset_f1s, max_mw_onset_2_f1, axes[2])
    
    # remove top border so labels are all visible
    # as well as right side border so it looks nice
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        
    # get legend for overall figure since its the same for both subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    
    plt.tight_layout(rect=[0,0,.95,1])
    if filtered:
        plt.savefig(f"./onset_v_self_report/{window_size}s/filtered_test_auroc_f1_onset_self_report_{window_size}s_win")
    else:
        plt.savefig(f"./onset_v_self_report/{window_size}s/unfiltered_test_auroc_f1_onset_self_report_{window_size}s_win")
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
    Plot predicted probabilities for the positive class as a function of relative 
    time. For each model type, a figure will be created with three subplots: 
    one to show predicted probabilities of control events, one to show 
    predicted probabilities when the true event was MW_onset, and one to show 
    predicted probabilities when the true event was self_report. For each subplot, 
    there are three lines. One line shows aggregated (mean over matching relative_times) 
    predicted probabilities of the positive class from models trained to 
    distinguish between MW_onset and control, while the other shows aggregated 
    (mean over matching relative_times) predicted probabilities of the positive 
    class from models trained to distinguish between self_report and control.
    Lastly, the third line shows aggregated predicted probabilities of the positive class
    from models trained to distinguish between MW_onset_2 and control.
    For each subplot, the standard error over the aggregated probabilities is also shown.
    
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

    #print("Before modifying relative times in function:")
    #print(X_test_relative_times[true_labels_test_set == 2].unique())

    # adjust relative times to handle self_report offset (since the event center
    # for those is - 1/2 window size)
    # no longer remove offset - misleading
    """
    X_test_relative_times_adjusted = X_test_relative_times.copy()  # dont alter OG
    X_test_relative_times_adjusted.loc[true_labels_test_set == 2] += (0.5 * window_size)
    
    print("After modifying relative times in function:")
    print(X_test_relative_times_adjusted[true_labels_test_set == 2].unique())
    """

    # merge relative times and y test on index
    #metadata = pd.concat([X_test_relative_times_adjusted, true_labels_test_set], axis=1)
    metadata = pd.concat([X_test_relative_times, true_labels_test_set], axis=1)
    plot_types = ["Control", "MW_Onset", "self_report"]
    
    # create these subplots for each model type
    for model_name, results in test_results.items():
        fig, axes = plt.subplots(2,2, figsize =(15,15), sharex=True, sharey=True)
        fig.suptitle(f"{model_name}: Predicted Probabilities as a Function of Relative Time, {window_size}s Sliding Window")
        
        axes = axes.flatten()
        fig.delaxes(axes[3]) # get rid of the empty subplot
        # get predicted probs for both classifier types, retaining indices
        # since these are all probabilities for each classifier type from X_test, 
        # indices should match those in the full series of relative times
        #MW_onset_probs = pd.Series(results["MW_onset_y_scores"], index = X_test_relative_times_adjusted.index)
        #self_report_probs = pd.Series(results["self_report_y_scores"], index = X_test_relative_times_adjusted.index)
        MW_onset_probs = pd.Series(results["MW_onset_y_scores"])#, index = X_test_relative_times.index)
        self_report_probs = pd.Series(results["self_report_y_scores"])#, index = X_test_relative_times.index)
        MW_onset_2_probs = pd.Series(results["MW_onset_2_y_scores"])

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
            
            MW_onset_2_mean_probs = subset.groupby("relative_time").apply(lambda x: np.mean(MW_onset_2_probs[x.index]))
            MW_onset_2_probs_se = subset.groupby("relative_time").apply(lambda x: MW_onset_2_probs[x.index].sem())
            
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
            
            # MW onset 2
            ax.plot(MW_onset_2_mean_probs.index, MW_onset_2_mean_probs, label="MW_onset 2 Classifier")
            ax.fill_between(MW_onset_2_mean_probs.index,
                            MW_onset_2_mean_probs - MW_onset_2_probs_se,
                            MW_onset_2_mean_probs + MW_onset_2_probs_se,
                            alpha = .4)
            
            # subplot details
            ax.set_title(f"{plot_type}")
            ax.set_xlabel("Window Midpoint Relative to Event Onset (s)")
            if subplot_idx %2 == 0:
                ax.set_ylabel("Predicted Probability of Positive Class")
            # dashed line for event onset
            # set dynamically for self report
            if plot_type == "self_report":
                ax.axvline(-.5 * window_size, color="black", linestyle = "--", label = "Event Onset")
            # at x=0 otherwise
            else:
                ax.axvline(0, color="black", linestyle = "--", label="Event Onset")
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(f"./onset_v_self_report/{window_size}s/{model_name}_probabilities")
        plt.show()
        
def plot_raw_scores(full_set_relative_times, true_labels_full_set, window_size, raw_scores):
    """
    Plot raw scores as a function of relative time. For each linear model type, a 
    figure will be created with three subplots: one to show raw scores of control 
    events, one to show raw scores when the true event was MW_onset, and one 
    to show raw scores when the true event was self_report. For each subplot, 
    there are three lines. One line shows aggregated (mean over matching relative_times) 
    raw scores from models trained to distinguish between MW_onset and control, 
    one shows raw scores from models trained to distinguish between MW_onset_2 and
    control, while the other shows aggregated (mean over matching relative_times) raw scores 
    from models trained to distinguish between self_report and control. For 
    each subplot, the standard error over the aggregated raw scores is also shown.
    
    Parameters
    ----------
    full_set_relative_times : Series
        Relative times of samples in the full dataset, with indices matching X (full dataset features).
    true_labels_full_set : Series
        True labels for the full dataset. This should not be the same as the labels used for 
        evaluation that contain a 1 for MW_Onset OR self_report. Instead, this should
        have 1 for MW_onset, 2 for self_report, and 0 for control.
    window_size : Int.
        The size of the sliding window used throughout the pipeline.
    raw_scores : Dict.
        Raw scores for each model type (mw_onset or self report) for each linear model.
        These are stored in a nested dictionary where first level keys are model
        names and second level keys are "self_report_raw_scores" and "MW_onset_raw_scores".
        Raw scores are the dot product of model weights and features such that there
        is one score for each sample.

    Returns
    -------
    None.

    """


    # merge relative times and y test on index
    #metadata = pd.concat([X_test_relative_times_adjusted, true_labels_test_set], axis=1)
    metadata = pd.concat([full_set_relative_times, true_labels_full_set], axis=1)
    plot_types = ["Control", "MW_Onset", "self_report"]
    
    # create these subplots for each model type
    for model_name, results in raw_scores.items():
        fig, axes = plt.subplots(2,2, figsize =(15,15), sharex=True, sharey=True)
        fig.suptitle(f"{model_name}: Raw Scores as a Function of Relative Time, {window_size}s Sliding Window")
        
        axes = axes.flatten()
        fig.delaxes(axes[3]) # get rid of the empty subplot
        # get predicted probs for both classifier types, retaining indices
        # since these are all probabilities for each classifier type from X_test, 
        # indices should match those in the full series of relative times
        #MW_onset_probs = pd.Series(results["MW_onset_y_scores"], index = X_test_relative_times_adjusted.index)
        #self_report_probs = pd.Series(results["self_report_y_scores"], index = X_test_relative_times_adjusted.index)
        MW_onset_scores = pd.Series(results["MW_onset_raw_scores"])
        self_report_scores = pd.Series(results["self_report_raw_scores"])
        MW_onset_2_scores = pd.Series(results["MW_onset_2_raw_scores"])

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
            MW_onset_mean_scores = subset.groupby("relative_time").apply(lambda x: np.mean(MW_onset_scores[x.index]))
            MW_onset_scores_se = subset.groupby("relative_time").apply(lambda x: MW_onset_scores[x.index].sem())
            
            self_report_mean_scores = subset.groupby("relative_time").apply(lambda x: np.mean(self_report_scores[x.index]))
            self_report_scores_se = subset.groupby("relative_time").apply(lambda x: self_report_scores[x.index].sem())
            
            MW_onset_2_mean_scores = subset.groupby("relative_time").apply(lambda x: np.mean(MW_onset_2_scores[x.index]))
            MW_onset_2_scores_se = subset.groupby("relative_time").apply(lambda x: MW_onset_2_scores[x.index].sem())
            # plot lines and SE- index of mean probs is relative times bc of groupby
            # MW onset
            ax.plot(MW_onset_mean_scores.index, MW_onset_mean_scores, label="MW_onset Classifier")
            ax.fill_between(MW_onset_mean_scores.index,
                            MW_onset_mean_scores - MW_onset_scores_se,
                            MW_onset_mean_scores + MW_onset_scores_se,
                            alpha = .4)
            # self report
            ax.plot(self_report_mean_scores.index, self_report_mean_scores, label="self_report Classifier")
            ax.fill_between(self_report_mean_scores.index,
                            self_report_mean_scores - self_report_scores_se,
                            self_report_mean_scores + self_report_scores_se,
                            alpha = .4)
            
            #mw onset 2
            ax.plot(MW_onset_2_mean_scores.index, MW_onset_2_mean_scores, label="MW_onset 2 Classifier")
            ax.fill_between(MW_onset_2_mean_scores.index,
                            MW_onset_2_mean_scores - MW_onset_2_scores_se,
                            MW_onset_2_mean_scores + MW_onset_2_scores_se,
                            alpha = .4)
            
            # subplot details
            ax.set_title(f"{plot_type}")
            ax.set_xlabel("Window Midpoint Relative to Event Onset (s)")
            if subplot_idx %2 == 0:
                ax.set_ylabel("Raw Score")
            # dashed line for event onset
            # set dynamically for self report
            if plot_type == "self_report":
                ax.axvline(-.5 * window_size, color="black", linestyle = "--", label = "Event Onset")
            # at x=0 otherwise
            else:
                ax.axvline(0, color="black", linestyle = "--", label="Event Onset")
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(f"./onset_v_self_report/{window_size}s/{model_name}_raw_scores")
        plt.show()
        
def plot_rf_feat_importance(importances, columns, window_size, classifier_type):
    """
    Plots random forest feature importance. The resulting plot is saved. Code 
    adapted from HS.

    Parameters
    ----------
    importances : 
    columns : Index
        Columns/ features as ordered in the training/ testing data.
    window_size : Int.
        The size of the sliding window (s.)
    classifier_type : Str.
        The classifier type: mw_onset or self_report. Represents what data the
        model was trained on.


    Returns
    -------
    None.

    """

    indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order

    # Plot feature importance
    plt.figure(figsize=(12, 10))
    plt.title(f"Random Forest Feature Importance: {classifier_type} vs. control Classifier, {window_size}s Sliding Window")
    plt.bar(range(len(columns)), importances[indices], align="center")
    plt.xticks(range(len(columns)), [columns[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.savefig(f"./onset_v_self_report/{window_size}s/rf_feat_importance_{classifier_type}")
    plt.show()

def get_raw_scores(mw_models, sr_models, mw_2_models, X_mw, X_mw2, X_sr):
    """
    Finds raw scores for each model type (mw onset, mw onset 2, or self report), 
    for each linear model. Raw scores are defined as Z from the forward model where Z
    is the dot product of features and model weights such that there is
    one raw score for each sample.

    Parameters
    ----------
    mw_models : Dict.
        Dictionary holding models trained to distinguish between MW onset and control.
    sr_models : Dict.
        Dictionary holding models trained to distinguish between self report and control.
    mw2_models : Dict.
        Dictionary holding models trained to distinguish between MW onset at relative
        time 2 and control.
    X_sr : DataFrame.
        Features for the full dataset. If PCA_flag was True for training, this
        set has been transformed using the same PCA model as the train set for 
        self_report. Otherwise it is identical to X_mw and X_mw2.
    X_mw : DataFrame.
        Features for the full dataset. If PCA_flag was True for training, this
        set has been transformed using the same PCA model as the train set for 
        mw_onset. Otherwise it is identical to X_sr and X_mw2.
    X_mw2 : DataFrame.
        Features for the full dataset. If PCA_flag was True for training, this 
        set has been transformed using the same PCA model as the train set for 
        mw_onset_2. Otherwise it is identical to X_mw and X_sr.
    

    Returns
    -------
    results : Dict.
        Nested dictionary holding raw scores for each linear model (SVM, logistic 
        regression, LDA), for each model type (mw onset vs. control and self 
        report vs. control). First level keys are model names while second level 
        keys are mw_onset_raw_scores and self_report_raw_scores. 

    """
    # these raw scores can later be used for calculating feature importance or
    # for plotting different event behavior as fntn of relative time
    linear_models = ["Support Vector Machine", "Logistic Regression", "Linear Discriminant Analysis"]
    # return nested dict, 1x level keys model names, 2nd mw_onset or self_report
    #for each linear model (mw and sr), calc forward model
    
    # initialize results dict - nested with first level keys as model names, second level keys
    # self report raw scores or mw onset raw scores so scores for each model type are held for each model name (linear only)
    results = {model_name:{
        "self_report_raw_scores" : None,
        "MW_onset_raw_scores" : None,
        "MW_onset_2_raw_scores" : None} for model_name in linear_models} # only getting results for linear
    
    # mw_onset loop
    for model_name, model in mw_models.items():
        # loop through all models, but only execute logic if linear
        if model_name in linear_models:
            #get z, add to right part of results dict
            # get the weights
            if hasattr(model, "coef_"):
                w = model.coef_.flatten() # list, one entry for each feature
                # find Z - dot prod of X_test and weights
                Z = np.dot(X_mw, w)  # array, one entry for each sample: these are raw scores
                # add to dict
                results[model_name]["MW_onset_raw_scores"] = pd.Series(Z, index=X_mw.index) 
            else:
                print("The specified model doesn't have the coef_ attribute...")
                
    # self report loop
    for model_name, model in sr_models.items():
        # loop through all models, but only execute logic if linear
        if model_name in linear_models:
            #get z, add to right part of results dict
            # get the weights
            if hasattr(model, "coef_"):
                w = model.coef_.flatten() # list, one entry for each feature
                # find Z - dot prod of X_test and weights
                Z = np.dot(X_sr, w)  # array, one entry for each sample: these are raw scores
                # add to dict
                results[model_name]["self_report_raw_scores"] = pd.Series(Z, index = X_sr.index) 
            else:
                print("The specified model doesn't have the coef_ attribute...")
                
    # mw_onset 2 loop
    for model_name, model in mw_2_models.items():
        # loop through all models, but only execute logic if linear
        if model_name in linear_models:
            #get z, add to right part of results dict
            # get the weights
            if hasattr(model, "coef_"):
                w = model.coef_.flatten() # list, one entry for each feature
                # find Z - dot prod of X_test and weights
                Z = np.dot(X_mw2, w)  # array, one entry for each sample: these are raw scores
                # add to dict
                results[model_name]["MW_onset_2_raw_scores"] = pd.Series(Z, index=X_mw2.index) 
            else:
                print("The specified model doesn't have the coef_ attribute...")
                
    return results

def plot_forward_feature_importance(raw_scores, columns, window_size, X_mw, X_mw2, X_sr):
    """
    Calculate and plot forward model feature importance for each linear model,
    for each model type (mw_onset vs. control or self report vs. control). Creates
    one plot for each linear model with two subplots: one for the mw_onset vs. control
    classifier and one for the self_report vs. control classifier.

    Parameters
    ----------
    raw_scores : Dict.
        Nested dictionary holding raw scores for each linear model (SVM, logistic 
        regression, LDA), for each model type (mw onset vs. control and self 
        report vs. control). First level keys are model names while second level 
        keys are mw_onset_raw_scores and self_report_raw_scores. 
    columns : Index
        Columns/ features as ordered in the training/ testing data.
    window_size : Int.
        The size of the sliding window for the data currently being processed.
    X_mw : DataFrame.
        Features for the full dataset. If PCA_flag was True for training, this 
        set has been transformed using the same PCA model as the train set for 
        mw_onset. Otherwise it is identical to X_sr and X_mw2.
    X_mw2 : DataFrame.
        Features for the full dataset. If PCA_flag was True for training, this 
        set has been transformed using the same PCA model as the train set for 
        mw_onset_2. Otherwise it is identical to X_mw and X_sr.
    X_sr : DataFrame.
        Features for the full dataset. If PCA_flag was True for training, this 
        set has been transformed using the same PCA model as the train set for 
        self_report. Otherwise it is identical to X_mw and X_mw2.


    Returns
    -------
    None.
    
    """
    
    # calculate forward model feature importance and plot for each linear model, for each model type
    # one plot for each model feature importance with subplots for mw onset and sr model types
    
    
    for model_name, scores in raw_scores.items():
        Z_mw = scores["MW_onset_raw_scores"]
        Z_sr = scores["self_report_raw_scores"]
        Z_mw2 = scores["MW_onset_2_raw_scores"]
        
        # find A for mw model type
        A_numerator_mw = np.dot(X_mw.T, Z_mw) # transpose X so inner dimensions match for mult.
        A_denom_mw = np.dot(Z_mw.T, Z_mw)
        A_mw = A_numerator_mw / A_denom_mw # now A has one importance value for each feature
        
        # find A for mw 2 model type
        A_numerator_mw2 = np.dot(X_mw2.T, Z_mw2) # transpose X so inner dimensions match for mult.
        A_denom_mw2 = np.dot(Z_mw2.T, Z_mw2)
        A_mw2 = A_numerator_mw2 / A_denom_mw2 # now A has one importance value for each feature
        
        # find A for sr model type
        A_numerator_sr = np.dot(X_sr.T, Z_sr) # transpose X so inner dimensions match for mult.
        A_denom_sr = np.dot(Z_sr.T, Z_sr)
        A_sr = A_numerator_sr / A_denom_sr # now A has one importance value for each feature
        
        feature_names = columns
            
        # plot
        # Get feature importances for each model type
        importances_mw = np.abs(A_mw)
        importances_sr = np.abs(A_sr)
        importances_mw2 = np.abs(A_mw2)
    
        indices_mw = np.argsort(importances_mw)[::-1]  # Sort feature importances in descending order
        indices_sr = np.argsort(importances_sr)[::-1]  # Sort feature importances in descending order
        indices_mw2 = np.argsort(importances_mw2)[::-1] 
        
        fig, axes = plt.subplots(3,1,figsize=(12,10))
    
        # Plot feature importance
        # mw onset
        axes[0].bar(range(len(feature_names)), importances_mw[indices_mw])
        axes[0].set_xticks(range(len(feature_names)))
        axes[0].set_xticklabels([feature_names[i] for i in indices_mw], rotation=45, ha="right")
        axes[0].set_ylabel("Importance")
        axes[0].set_title("MW_Onset vs. Control")
        
        # self report
        axes[1].bar(range(len(feature_names)), importances_sr[indices_sr])
        axes[1].set_xticks(range(len(feature_names)))
        axes[1].set_xticklabels([feature_names[i] for i in indices_sr], rotation=45, ha="right")
        axes[1].set_ylabel("Importance")
        axes[1].set_title("Self-Report vs. Control")
        
        # mw onset 2
        axes[2].bar(range(len(feature_names)), importances_mw2[indices_mw2])
        axes[2].set_xticks(range(len(feature_names)))
        axes[2].set_xticklabels([feature_names[i] for i in indices_mw2], rotation=45, ha="right")
        axes[2].set_ylabel("Importance")
        axes[2].set_title("MW_Onset 2 vs. Control")
        
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle(f"Forward Model Feature Importance: {model_name} {window_size}s Sliding Window")
        plt.savefig(f"onset_v_self_report/{window_size}s/{model_name}_forward_feat_importance.png") 
        plt.show()
        
def predictor_hist(raw_scores, true_labels_full_set, classifier_type, window_size):
    """
    
    Plot and saves a histogram of the predictor variable for each model. One figure
    is created with a subplot for each model type. Uses raw scores rather than
    predicted probabilities, so plots are only made for linear models. The
    data is separated into two distributions based on whether the ground truth label 
    indicates MW or not. Medians of both distributions are displayed, and the model's
    decision boundary is marked. 
    
    Parameters:
    raw_scores : Dict.
        Nested dictionary holding raw scores for each linear model (SVM, logistic 
        regression, LDA), for each model type (mw onset vs. control and self 
        report vs. control). First level keys are model names while second level 
        keys are mw_onset_raw_scores and self_report_raw_scores. 
    true_labels_full_set : Series
        True labels for the full dataset. This should not be the same as the labels used for 
        evaluation that contain a 1 for MW_Onset OR self_report. Instead, this should
        have 1 for MW_onset, 2 for self_report, and 0 for control.
    classifier_type : Str.
        The classifier type: mw_onset or self_report. Represents what data the
        model was trained on.
    window_size : Int.
        The size of the sliding window for the data currently being processed.
    ---------
    Returns : 
        None.

    """
    # one figure for each model name with a subplot for each model type: mw onset, mw onset 2, or self report
    
    # in true labels, 1=mw, 2=sr, 0=ctl
    
    # set y scores keys and labels depending on classifier type
    # for labels, set the target class to 1, keep control as 0, set the non target class to 0
    if classifier_type == "mw_onset":
        raw_score_key = "MW_onset_raw_scores"
        suptitle_addition = "MW_onset vs. Control"
        y = true_labels_full_set.apply(lambda x: 1 if x==1 else 0) # 1 if mw, else 0
    elif classifier_type == "self_report":
        raw_score_key = "self_report_raw_scores"
        suptitle_addition = "self_report vs. Control"
        y = true_labels_full_set.apply(lambda x: 1 if x==2 else 0) #1 if sr, else 0
    elif classifier_type == "mw_onset_2":
        raw_score_key = "MW_onset_2_raw_scores"
        suptitle_addition = "MW_onset 2 vs. Control"
        y = true_labels_full_set.apply(lambda x: 1 if x==1 else 0) # 1 if mw, else 0

    fig, axes = plt.subplots(3,1,figsize=(15,15))
    fig.suptitle(f"Predictor Histograms, {window_size}s Sliding Window, {suptitle_addition}", fontsize=16)
    # flatten axes
    axes = np.array(axes).flatten()
    
    # get y_pred for each model- now raw scores

        
    for idx, (model_name, metrics) in enumerate(raw_scores.items()):

        scores = metrics[raw_score_key]
        labels = y

        scores = np.array(scores)
        labels = np.array(labels)
        # classify preds by labels
        preds_0 = scores[labels == 0]
        preds_1 = scores[labels == 1]
        
        ax = axes[idx]
        # plot KDEs
        #sns.kdeplot(preds_0, ax=ax, label="Actually Control", fill=True, color = "orange")
        #sns.kdeplot(preds_1, ax=ax, label = "Actually MW", fill=True, color = "blue")
        ax.hist(preds_0, label="Actually not MW", color = "orange", alpha=.5, bins=20)
        ax.hist(preds_1, label="Actually MW", color = "blue", alpha=.5, bins=20)
        
        # median markers
        ax.axvline(np.median(preds_0), color = "orange", linestyle = "-", label = "Median (Control)")
        ax.axvline(np.median(preds_1), color = "blue", linestyle = "-", label = "Median (MW)")
        
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Raw Score")
        ax.set_ylabel("Density")
        ax.legend()
        
    plt.tight_layout()  
    plt.savefig(f"onset_v_self_report/{window_size}s/Predictor_hist_{classifier_type}.png")
    
    
#%%
    

window_size = 5 # options are two and five. 2 has less imbalance
random_state = 42 # shouldn't need to change this
SMOTE_flag = True # set to True to use SMOTE (synthetic minority over-sampling technique) to achieve more balanced class dist.
undersample_flag = False # set to True to use undersampling to achieve more balanced class dist. 
PCA_flag = True
PCA_threshold = .9 # for PCA
# DO NOT SET BOTH UNDERSAMPLE_FLAG AND SMOTE_FLAG TO TRUE AT THE SAME TIME THAT WOULD BE WEIRD/ ERROR WILL BE THROWN

assert not (SMOTE_flag == True and undersample_flag == True), "Error: cannot use SMOTE and undersampling at the same time this way. Change one to False or adapt the pipeline."
    

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

# apply same filter logic to a filtered x test for alternative performance metric calculations
X_test_filtered = X_test[
    ((X_test["label"] == "control") & (X_test["relative_time"] == 0)) |
    ((X_test["label"] == "self_report") & (X_test["relative_time"] == (-.5 * window_size))) |
    ((X_test["label"] == "MW_onset") & (X_test["relative_time"] == 0)) |
    ((X_test["label"] == "MW_onset") & (X_test["relative_time"] == 2)) | # add condition for mw onset + 2
    ((X_test["label"] == "control") & (X_test["relative_time"] == 2)) |
    ((X_test["label"] == "control") & (X_test["relative_time"] == (-.5 * window_size)))
    ]
# sanity checks.. should have all three label types and only 0 as relative time 
# val for each group except self_report, which should have only -1 or -2.5 depending on window size.
print("unique vals for label column:", X_train["label"].unique())
print("unique vals for relative time, groupedby label: ")
print(X_train.groupby("label")["relative_time"].unique())

# adjust y_train accordingly so that it matches full X_train (same rows dropped)
y_train = y_train.loc[X_train.index]

# do the same for y test filtered
y_test_filtered = y_test.loc[X_test_filtered.index]


# drop rel time from each train set after creating

# create X_train_MW_onset: only keep rows where label isn't self report and relative time is 0
# then drop label & page. Also create groups for CV

# do the same for the filtered x test sets

X_train_MW_onset = X_train[(X_train["label"] != "self_report") & (X_train["relative_time"] == 0)]
X_train_MW_onset = X_train_MW_onset.copy()
X_train_MW_onset.drop(columns=["label", "page", "relative_time"], inplace=True)

X_test_filtered_MW_onset = X_test_filtered[(X_test_filtered["label"] != "self_report") & (X_test_filtered["relative_time"] == 0)]
X_test_filtered_MW_onset = X_test_filtered_MW_onset.copy()
X_test_filtered_MW_onset.drop(columns=["label", "page", "relative_time"], inplace=True)

# create X_train_MW_onset_2
X_train_MW_onset_2 = X_train[(X_train["label"] != "self_report") & (X_train["relative_time"] == 2)]
X_train_MW_onset_2 = X_train_MW_onset_2.copy()
X_train_MW_onset_2.drop(columns=["label", "page", "relative_time"], inplace=True)

X_test_filtered_MW_onset_2 = X_test_filtered[(X_test_filtered["label"] != "self_report") & (X_test_filtered["relative_time"] == 2)]
X_test_filtered_MW_onset_2 = X_test_filtered_MW_onset_2.copy()
X_test_filtered_MW_onset_2.drop(columns=["label", "page", "relative_time"], inplace=True)

# create X_train_self_report: drop rows where label = mw onset (retain self report and control), then drop label & page
X_train_self_report = X_train[(X_train["label"] != "MW_onset") & (X_train["relative_time"] == (-.5 * window_size))]
X_train_self_report = X_train_self_report.copy()
X_train_self_report.drop(columns=["label", "page", "relative_time"], inplace=True) 

X_test_filtered_self_report = X_test_filtered[(X_test_filtered["label"] != "MW_onset") & (X_test_filtered["relative_time"] == (-.5 * window_size))]
X_test_filtered_self_report = X_test_filtered_self_report.copy()
X_test_filtered_self_report.drop(columns=["label", "page", "relative_time"], inplace=True)

# drop relative time from X_train and X_test now
X_train_relative_times = X_train["relative_time"].copy()
X_train.drop(columns=["relative_time"], inplace=True)
X_test_relative_times = X_test["relative_time"].copy()
#X_test_relative_times.reset_index(inplace=True, drop=True)
X_test.drop(columns=["relative_time"], inplace=True)
X_test_filtered_relative_times = X_test_filtered["relative_time"].copy() # now we have relative times for X_test_filtered saved
# each x_test specific to a classifier has been further narrowed down by relative time and label




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
    print("Indexes for X train relative times and y train match!")
else:
    print("Indexes do not match (X train relative times and y train).")
    
# same for filtered test
if X_test_filtered_relative_times.index.equals(y_test_filtered.index):
    print("Indexes for X test filtered relative times and y test filtered match!")
else:
    print("Indexes do not match (X test filtered relative times and y test filtered).")
    
# at this stage there are only mw events for rel time 2 in y train


# create y_train_MW_onset: 0 for control, 1 for MW_onset when rel time = 0. Rows where label = self_report dropped
y_train_MW_onset = y_train[X_train_relative_times == 0]
y_train_MW_onset = y_train_MW_onset[y_train_MW_onset != "self_report"]
y_train_MW_onset = y_train_MW_onset.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# same for y test filtered
y_test_filtered_MW_onset = y_test_filtered[X_test_filtered_relative_times == 0]
y_test_filtered_MW_onset = y_test_filtered_MW_onset[y_test_filtered_MW_onset != "self_report"]
y_test_filtered_MW_onset = y_test_filtered_MW_onset.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_MW_onset_2: : 0 for control, 1 for MW_onset when rel time = 2. Rows where label = self_report dropped
y_train_MW_onset_2 = y_train[X_train_relative_times == 2]
y_train_MW_onset_2 = y_train_MW_onset_2[y_train_MW_onset_2 != "self_report"]
y_train_MW_onset_2 = y_train_MW_onset_2.apply(lambda x: 1 if x in ["MW_onset"] else 0)

y_test_filtered_MW_onset_2 = y_test_filtered[X_test_filtered_relative_times == 2]
y_test_filtered_MW_onset_2 = y_test_filtered_MW_onset_2[y_test_filtered_MW_onset_2 != "self_report"]
y_test_filtered_MW_onset_2 = y_test_filtered_MW_onset_2.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_self_report: 0 for control, 1 for self_report. Rows where label = MW_onset dropped
y_train_self_report = y_train[X_train_relative_times == (-.5 * window_size)]
y_train_self_report = y_train_self_report[y_train_self_report != "MW_onset"]
y_train_self_report = y_train_self_report.apply(lambda x: 1 if x in ["self_report"] else 0)

y_test_filtered_self_report = y_test_filtered[X_test_filtered_relative_times == (-.5 * window_size)]
y_test_filtered_self_report = y_test_filtered_self_report[y_test_filtered_self_report != "MW_onset"]
y_test_filtered_self_report = y_test_filtered_self_report.apply(lambda x: 1 if x in ["self_report"] else 0)

# create a label set where all original labels are retained for plotting later
# this is not to be used for any training or testing, just as a source of truth for plotting

# this time we do this for full y set (train and test)
y_copy = y.copy()
true_labels_full_set = y_copy.apply(lambda x: 1 if x == "MW_onset" else (2 if x == "self_report" else 0))

# binarize y_test: 0 for control, 1 for self_report OR mw_onset
y_test = y_test.apply(lambda x: 1 if x in ["MW_onset", "self_report"] else 0)

# filter full dataset X so columns match train sets - drop label, page, relative_time after saving relative time separately
full_set_relative_times = X["relative_time"]
X = X.copy()
X.drop(columns=["label", "page", "relative_time"], inplace=True)


if full_set_relative_times.index.equals(true_labels_full_set.index):
    print("Indexes for full set relative times and true labels full set match!")
else:
    print("Indexes do not match (full set relative times and full set true labels).")

# define models
if window_size == 2:
    self_report_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=10), 
            'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.1), 
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            'Random Forest': RandomForestClassifier(random_state = random_state, n_estimators = 200),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(), 
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(),
            #'XGBoost': XGBClassifier(random_state = random_state)
        }
    
    MW_onset_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=.01),
            'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=10),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 20, n_estimators = 100),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(),
            #'XGBoost': XGBClassifier(random_state = random_state)
        }
    
    MW_onset_2_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=.00001),
            'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.01),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 20, n_estimators = 10),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(),
            #'XGBoost': XGBClassifier(random_state = random_state)
        }
    
if window_size == 5:
    self_report_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=10, solver="saga"), 
            'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=10), 
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 200),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME.R", learning_rate = .1, n_estimators = 100, ),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(), 
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(var_smoothing = .000000001),
            #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=.5, learning_rate = .001, max_depth = 3, n_estimators = 200, subsample=.5)
        }
    
    MW_onset_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=.0001),
            'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.001),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 200),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME", learning_rate=.1, n_estimators=50),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(var_smoothing = .000000001),
            #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=1, learning_rate=.001, max_depth=3, n_estimators=100, subsample=.5)
        }
    
    MW_onset_2_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=.0001),
            'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.001),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 100),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME", learning_rate=.1, n_estimators=100),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(var_smoothing= .000000001),
            #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=1, learning_rate=.01, max_depth=3, n_estimators=100, subsample=1)
        }

# sanity checks
#print(f"MW_Onset Features: {X_train_MW_onset.columns}")
#print(f"Self Report Features: {X_train_self_report.columns}")

# train self report vs. control

# first, print class distributions
if SMOTE_flag:
    print("Prior to SMOTE: ")
    
if undersample_flag:
    print("Prior to undersampling: ")
    
print("MW Onset Train Set Class Distribution ")

not_mw = (y_train_MW_onset == 0).sum()
mw = (y_train_MW_onset == 1).sum()
print(f"mw: {mw}, not mw {not_mw}")

print("MW Onset 2 Train Set Class Distribution ")

not_mw = (y_train_MW_onset_2 == 0).sum()
mw = (y_train_MW_onset_2 == 1).sum()
print(f"mw: {mw}, not mw {not_mw}")
                                               
print("Self Report Train Set Class Distribution")
not_mw = (y_train_self_report == 0).sum()
mw = (y_train_self_report == 1).sum()
print(f"mw: {mw}, not mw {not_mw}")

if SMOTE_flag:
    # SMOTE - oversample minority class individually for each train set (mw onset, self report, mw onset 2)
    # must be done separately because each has a different distribution.. could try oversampling full train set 
    # if desired, but self report and mw onset would be treated as the same minority class
    
    # initialize SMOTE objects for each dataset
    # sampling strat .8 to avoid overfitting to the synthetic train set as seen when auto strategy used/ completely balanced
    oversample_mw = SMOTE(random_state = random_state)
    oversample_mw2 = SMOTE(random_state = random_state)
    oversample_sr = SMOTE(random_state = random_state)
    
    X_train_MW_onset, y_train_MW_onset = oversample_mw.fit_resample(X_train_MW_onset, y_train_MW_onset)
    X_train_self_report, y_train_self_report = oversample_sr.fit_resample(X_train_self_report, y_train_self_report)
    X_train_MW_onset_2, y_train_MW_onset_2 = oversample_mw2.fit_resample(X_train_MW_onset_2, y_train_MW_onset_2)
    
    print("After SMOTE: ")
    print("MW Onset Train Set Class Distribution ")
    
    not_mw = (y_train_MW_onset == 0).sum()
    mw = (y_train_MW_onset == 1).sum()
    print(f"mw: {mw}, not mw {not_mw}")
    
    print("MW Onset 2 Train Set Class Distribution ")
    
    not_mw = (y_train_MW_onset_2 == 0).sum()
    mw = (y_train_MW_onset_2 == 1).sum()
    print(f"mw: {mw}, not mw {not_mw}")
                                                   
    print("Self Report Train Set Class Distribution")
    not_mw = (y_train_self_report == 0).sum()
    mw = (y_train_self_report == 1).sum()
    print(f"mw: {mw}, not mw {not_mw}")
    
if undersample_flag:
    # initialize undersampling objects for each dataset
    undersample_mw = RandomUnderSampler(random_state = random_state)
    undersample_mw2 = RandomUnderSampler(random_state = random_state)
    undersample_sr = RandomUnderSampler(random_state = random_state)
    
    X_train_MW_onset, y_train_MW_onset = undersample_mw.fit_resample(X_train_MW_onset, y_train_MW_onset)
    X_train_self_report, y_train_self_report = undersample_sr.fit_resample(X_train_self_report, y_train_self_report)
    X_train_MW_onset_2, y_train_MW_onset_2 = undersample_mw2.fit_resample(X_train_MW_onset_2, y_train_MW_onset_2)
    
    print("After undersampling: ")
    print("MW Onset Train Set Class Distribution ")
    
    not_mw = (y_train_MW_onset == 0).sum()
    mw = (y_train_MW_onset == 1).sum()
    print(f"mw: {mw}, not mw {not_mw}")
    
    print("MW Onset 2 Train Set Class Distribution ")
    
    not_mw = (y_train_MW_onset_2 == 0).sum()
    mw = (y_train_MW_onset_2 == 1).sum()
    print(f"mw: {mw}, not mw {not_mw}")
                                                   
    print("Self Report Train Set Class Distribution")
    not_mw = (y_train_self_report == 0).sum()
    mw = (y_train_self_report == 1).sum()
    print(f"mw: {mw}, not mw {not_mw}")


#train self_report vs control
# reset indices to ensure alignment
X_train_self_report.reset_index(inplace=True, drop=True)
y_train_self_report = y_train_self_report.reset_index(drop=True)

self_report_models, self_report_results, X_test_sr, X_sr, self_report_filtered_results, X_test_filtered_self_report = train_models(X_train_self_report, 
                                                                                                      y_train_self_report,
                                                                                                      X_test, X_test_filtered_self_report,
                                                                                                      self_report_models, PCA_flag,
                                                                                                      random_state,PCA_threshold, X)
        
# train MW_onset vs. control 
X_train_MW_onset.reset_index(inplace=True, drop=True)
y_train_MW_onset = y_train_MW_onset.reset_index(drop=True)
MW_onset_models, MW_onset_results, X_test_mw, X_mw, MW_onset_filtered_results, X_test_filtered_MW_onset = train_models(X_train_MW_onset,
                                                                                             y_train_MW_onset, X_test,
                                                                                             X_test_filtered_MW_onset,
                                                                                             MW_onset_models, PCA_flag,
                                                                                             random_state, PCA_threshold, X)
    
# train MW_onset_2 vs control
X_train_MW_onset_2.reset_index(inplace=True, drop=True)
y_train_MW_onset_2 = y_train_MW_onset_2.reset_index(drop=True)
MW_onset_2_models, MW_onset_2_results, X_test_mw2, X_mw2, MW_onset_2_filtered_results, X_test_filtered_MW_onset_2 = train_models(X_train_MW_onset_2,
                                                                                                     y_train_MW_onset_2, X_test,
                                                                                                     X_test_filtered_MW_onset_2,
                                                                                                     MW_onset_2_models, PCA_flag,
                                                                                                     random_state, PCA_threshold, X)


# evaluate self_report classifier on test/ holdout set
# combine MW_onset and self_report results so dict structure matches
test_results = {model_name:{
    "self_report_y_scores" : None,
    "MW_onset_y_scores" : None,
    "MW_onset_2_y_scores" : None} for model_name in self_report_models.keys()}

for model_name in self_report_models: # just loop through sr models, names are the same for mw onset
    # add to results
    test_results[model_name]["self_report_y_scores"] = pd.Series(self_report_results[model_name]["y_scores"], index = X_test_sr.index)
    test_results[model_name]["MW_onset_y_scores"] = pd.Series(MW_onset_results[model_name]["y_scores"], index = X_test_mw.index)
    test_results[model_name]["MW_onset_2_y_scores"] = pd.Series(MW_onset_2_results[model_name]["y_scores"], index = X_test_mw2.index)
    
test_results_filtered = {model_name:{
    "self_report_y_scores" : None,
    "MW_onset_y_scores" : None,
    "MW_onset_2_y_scores" : None} for model_name in self_report_models.keys()}

for model_name in self_report_models: # just loop through sr models, names are the same for mw onset
    # add to results
    test_results_filtered[model_name]["self_report_y_scores"] = pd.Series(self_report_filtered_results[model_name]["y_scores"], index = X_test_filtered_self_report.index)
    test_results_filtered[model_name]["MW_onset_y_scores"] = pd.Series(MW_onset_filtered_results[model_name]["y_scores"], index = X_test_filtered_MW_onset.index)
    test_results_filtered[model_name]["MW_onset_2_y_scores"] = pd.Series(MW_onset_2_filtered_results[model_name]["y_scores"], index = X_test_filtered_MW_onset_2.index)


# get aurocs and f1s on full test set - evaluate generalization
aurocs = get_AUROC(y_test, self_report_models, MW_onset_models, MW_onset_2_models, test_results)
f1s = get_f1(y_test, X_test_sr, X_test_mw, X_test_mw2, self_report_models, MW_onset_models, MW_onset_2_models)

# get auroccs and f1s on filtered test set - evaluate performance on designated relative times
MW_onset_filtered_aurocs = get_one_AUROC(y_test_filtered_MW_onset, MW_onset_models,
                                         test_results_filtered, "MW_onset")
MW_onset_2_filtered_aurocs = get_one_AUROC(y_test_filtered_MW_onset_2, MW_onset_2_models,
                                         test_results_filtered, "MW_onset_2")
self_report_filtered_aurocs = get_one_AUROC(y_test_filtered_self_report, self_report_models,
                                         test_results_filtered, "self_report")

# combine filtered aurocs into one dictionary matching unfiltered auroc dict
filtered_aurocs = {model_name:{
    "self_report_auroc" : None,
    "MW_onset_auroc" : None,
    "MW_onset_2_auroc" : None} for model_name in self_report_models.keys()}

for model_name in filtered_aurocs.keys():
    filtered_aurocs[model_name]["self_report_auroc"] = self_report_filtered_aurocs[model_name]
    filtered_aurocs[model_name]["MW_onset_auroc"] = MW_onset_filtered_aurocs[model_name]
    filtered_aurocs[model_name]["MW_onset_2_auroc"] = MW_onset_2_filtered_aurocs[model_name]

# get f1s
MW_onset_filtered_f1s = get_one_f1(y_test_filtered_MW_onset, X_test_filtered_MW_onset, MW_onset_models)
MW_onset_2_filtered_f1s = get_one_f1(y_test_filtered_MW_onset_2, X_test_filtered_MW_onset_2, MW_onset_2_models)
self_report_filtered_f1s = get_one_f1(y_test_filtered_self_report, X_test_filtered_self_report, self_report_models)

# combine filtered f1s into one dict
filtered_f1s = {model_name:{
    "self_report_f1" : None,
    "MW_onset_f1" : None,
    "MW_onset_2_f1" : None} for model_name in self_report_models.keys()}

for model_name in filtered_f1s.keys():
    filtered_f1s[model_name]["self_report_f1"] = self_report_filtered_f1s[model_name]
    filtered_f1s[model_name]["MW_onset_f1"] = MW_onset_filtered_f1s[model_name]
    filtered_f1s[model_name]["MW_onset_2_f1"] = MW_onset_2_filtered_f1s[model_name]

# display filtered auroc and f1 in grouped bar chart
plot_auroc_f1(filtered_aurocs, filtered_f1s, window_size, filtered=True)

# display auroc and f1 in grouped bar chart 
plot_auroc_f1(aurocs, f1s, window_size, filtered=False)

# plot probabilities from self_report on holdout set and mw_onset on holdout set
# feed this the true_labels_test_set so it has different indicators for self_report and mw_onset

#plot_probs(test_results, X_test_relative_times, true_labels_test_set, window_size, CV)

# only plot feature importance if PCA false, otherwise we don't have meaningful col names so not interesting
if PCA_flag == False:
    rf = MW_onset_models["Random Forest"]
    #print(rf.feature_importances_)
    columns = X_train_MW_onset.columns
    
    importances = rf.feature_importances_
    
    plot_rf_feat_importance(importances, columns, window_size, "MW_onset")
    
    
    rf2 = self_report_models["Random Forest"]
    #print(rf2.feature_importances_)
    
    columns = X_train_self_report.columns
    plot_rf_feat_importance(rf2.feature_importances_, columns, window_size, "self_report")
    
    rf3 = MW_onset_2_models["Random Forest"]
    #print(rf2.feature_importances_)
    
    columns = X_train_self_report.columns
    plot_rf_feat_importance(rf3.feature_importances_, columns, window_size, "MW_onset_2")

raw_scores = get_raw_scores(MW_onset_models, self_report_models, MW_onset_2_models, X_mw, X_mw2, X_sr)
plot_raw_scores(full_set_relative_times, true_labels_full_set, window_size, raw_scores)

if PCA_flag == False:
    plot_forward_feature_importance(raw_scores, columns, window_size, X_mw, X_mw2, X_sr)

# plot predictor hists 

predictor_hist(raw_scores, true_labels_full_set, "mw_onset", window_size)
predictor_hist(raw_scores, true_labels_full_set, "self_report", window_size)
predictor_hist(raw_scores, true_labels_full_set, "mw_onset_2", window_size)
