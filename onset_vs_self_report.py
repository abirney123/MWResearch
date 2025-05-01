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
        
3/24 Note: Re-ran grid search, looking for optimal values for PCA and SMOTE as well.
* More often than not, grid search shows models perform better without SMOTE. However.
not using it results in failure to converge for some models and much lower F1.
* PCA on/ off and thresholds are currently tuned to be optimal for the 5s window
* Hyperparams are tuned for mw2_target = 2, 2.5, and 5. Other values will produce an error.
* Hyperparams have been tuned for the 5s windowed data. Running the 2s windowed data
will not result in an error, but hyperparameters have not been optimized for this.

4/14 Note: MW Onset events within 1.5 sec of self report have been filtered out.
This is true for 2 and 5s windowed data. Motivation: ensure the MW2.5 classifier
isn't just picking up on characteristics of self report.

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
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
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
    X_test_filtered: DataFrame
        Features for testing with rows filtered to match the X_train set specific to
        this classifier type.
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
    full_feature_set_no_PCA_transform : DataFrame
        Full feature set with the transformation from the scaler for the train
        set applied, but without the PCA transformation applied. This is for
        forward model calculation.

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
    full_feature_set_no_PCA_transform = pd.DataFrame(scaler.transform(full_feature_set), index = full_feature_set.index, columns = full_feature_set.columns)
    
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
    
            
    return models, results, X_test, full_feature_set, filtered_results, X_test_filtered, full_feature_set_no_PCA_transform

def train_models_cv(X_train, y_train, X_test, y_test, models, PCA_flag, random_state, threshold):
    """
    Train the provided models without cross validation. Reduces the train and test
    sets with PCA according to the provided threshold if specified.
    Similar to train_models, but doesn't handle X_test_filtered or full_feature_set. 
    This is to simplify to support training during cross validation. All results are
    from a "filtered" test set here due to the way the logocv splits work.

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

    Returns
    -------
    models : Dictionary
        Trained models. Keys are model names.
    results : Dictionary
        Nested dictionary of test results. First level keys are model names 
        as specified in the models dictionary. Second level keys are "y_scores"
        and hold the y_scores for on the full, unfiltered test set for the model denoted by the
        first level key.

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

        #num_components = rat_idx 
        reduce_pca = PCA(n_components = PCA_threshold, random_state = random_state)
        # fit transform train
        X_train = pd.DataFrame(reduce_pca.fit_transform(X_train),index = X_train_indices)
        # transform test
        X_test = pd.DataFrame(reduce_pca.transform(X_test), index = X_test_indices)
    
    # compute y scores with test set scaled the same way as current train set 
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        # get y-scores
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:,1]
        else:
            print("LOGOCV: model doesn't have predict proba")
            
        # save the info
        results[model_name]["y_scores"] = y_scores
        results[model_name]["true_labels"] = y_test
    
            
    return models, results

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


def plot_auroc_f1(aurocs, f1s, window_size, mw2_target, no_mw2_flag, filtered):
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
    no_mw2_flag: Bool.
        Boolean flag specifying whether or not to plot info related to the 
        mw_onset vs. control classifier trained on relative time != 0.
    mw2_target: Int
        The relative time that the second mw_onset vs. control classifier was 
        trained on. For example, 2 or 5. 
    filtered: Bool
        A flag to designate whether the displayed auroc and f1 were computed 
        from the filtered test set or from the unfiltered test set.
    

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
    if no_mw2_flag == False:
        # side by side plots for MW_onset and self_report models
        fig, axes = plt.subplots(1,3, figsize = (17,9), sharey=True, sharex=True)
    else: # if not plotting mw2, don't plot related metrics, only two subplots
        fig, axes = plt.subplots(1,2, figsize = (17,9), sharey=True, sharex=True)
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
    
    if no_mw2_flag == False:
        mwo_2_auroc_bars = axes[2].bar(x-.2, MW_onset_2_aurocs, .4, label="AUROC", color = auroc_color)
        mwo_2_f1_bars = axes[2].bar(x+.2, MW_onset_2_f1s, .4, label="F1", color = f1_color)
        axes[2].set_title(f"MW_Onset {mw2_target} Models")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45, ha="right")
        axes[2].set_ylabel("Score")
    
    add_labels(mwo_auroc_bars, MW_onset_aurocs, max_mw_onset_auroc, axes[0])
    add_labels(mwo_f1_bars, MW_onset_f1s, max_mw_onset_f1, axes[0])
    add_labels(sr_auroc_bars, self_report_aurocs, max_self_report_auroc, axes[1])
    add_labels(sr_f1_bars, self_report_f1s, max_self_report_f1, axes[1])
    if no_mw2_flag == False:
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
    if no_mw2_flag == False:
        if filtered:
            plt.savefig(f"./onset_v_self_report/{window_size}s/filtered_test_auroc_f1_onset_self_report_{window_size}s_win_mw{mw2_target}.png")
        else:
            plt.savefig(f"./onset_v_self_report/{window_size}s/unfiltered_test_auroc_f1_onset_self_report_{window_size}s_win_mw{mw2_target}.png")
    else: # specify in plot name if no mw2, include conditional filtered logic
        if filtered:
            plt.savefig(f"./onset_v_self_report/{window_size}s/filtered_test_auroc_f1_onset_self_report_{window_size}s_win_no_mw2.png")
        else:
            plt.savefig(f"./onset_v_self_report/{window_size}s/unfiltered_test_auroc_f1_onset_self_report_{window_size}s_win_no_mw2.png")
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
        
def plot_raw_scores(full_set_relative_times, true_labels_full_set, window_size,
                    raw_scores, no_mw2_flag, mw2_target, trial_count_df, mw_color,
                    sr_color, mw2_color):
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
    no_mw2_flag: Bool.
        Boolean flag specifying whether or not to plot the classifier trained to 
        distinguish between mw_onset and control for the relative time NOT equal
        to 0.
    mw2_target : Int.
        The relative time that the second mw_onset vs. control classifier was 
        trained on. For example, 2 or 5. 
    mw_color : String, matplotlib CSS color name.
        The color to use for data related to mind wandering in this plot.
    sr_color : String, matplotlib CSS color name.
        The color to use for data related to self report in this plot.
    mw2_color : String, matplotlib CSS color name.
        The color to use for data related to the second mind wandering classifier
        in this plot. The second mind wandering classifier refers to the classifier
        trained on mw_onset events in windows centered at relative times later than time 0. 

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
        fig, axes = plt.subplots(1,3, figsize =(15,6), sharex=True, sharey=True)
        fig.suptitle(f"{model_name}: Raw Scores as a Function of Relative Time, {window_size}s Sliding Window", fontsize=16)
        
        axes = axes.flatten()
        #fig.delaxes(axes[3]) # get rid of the empty subplot
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
                num_trials = trial_count_df["MW_trials"].iloc[0] # pull out the scalar
            elif plot_type == "self_report":
                subset = metadata[metadata["label"] == 2]
                num_trials = trial_count_df["sr_trials"].iloc[0]
            else: # control
                subset = metadata[metadata["label"] == 0]
                num_trials = trial_count_df["ctl_trials"].iloc[0]
                

            
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
            ax.plot(MW_onset_mean_scores.index, MW_onset_mean_scores, label="MW_onset Classifier", color=mw_color)
            ax.fill_between(MW_onset_mean_scores.index,
                            MW_onset_mean_scores - MW_onset_scores_se,
                            MW_onset_mean_scores + MW_onset_scores_se,
                            alpha = .4, color = mw_color)
            # self report
            ax.plot(self_report_mean_scores.index, self_report_mean_scores, label="self_report Classifier", color=sr_color)
            ax.fill_between(self_report_mean_scores.index,
                            self_report_mean_scores - self_report_scores_se,
                            self_report_mean_scores + self_report_scores_se,
                            alpha = .4, color = sr_color)
            
            if no_mw2_flag == False: # only plot the second mw classifier if flag is set to False 
                #mw onset 2
                ax.plot(MW_onset_2_mean_scores.index, MW_onset_2_mean_scores, label=f"MW_onset {mw2_target} Classifier", color=mw2_color)
                ax.fill_between(MW_onset_2_mean_scores.index,
                                MW_onset_2_mean_scores - MW_onset_2_scores_se,
                                MW_onset_2_mean_scores + MW_onset_2_scores_se,
                                alpha = .4, color = mw2_color)
            
            # subplot details
            ax.set_title(f"{plot_type}: {num_trials} Trials", fontsize=14)
            ax.set_xlabel("Window Midpoint Relative to Event Onset (s)", fontsize=14)
            if subplot_idx == 0:
                ax.set_ylabel("Raw Score (Mean over Relative Time +/- Standard Error)", fontsize=14)
            # dashed line for event onset
            # set dynamically for self report
            """
            if plot_type == "self_report":
                ax.axvline(-.5 * window_size, color="black", linestyle = "--", label = "Event Onset")
            # at x=0 otherwise
            else:
                ax.axvline(0, color="black", linestyle = "--", label="Event Onset")
            """
            ax.axvline(0, color="black", linestyle = "--", label="Event Onset")
            ax.legend(fontsize=12)
            ax.grid(True)
            
        plt.tight_layout()
        if no_mw2_flag: # if no mw2, say that in filename
            plt.savefig(f"./onset_v_self_report/{window_size}s/{model_name}_raw_scores_no_mw2.png")
        else:
            plt.savefig(f"./onset_v_self_report/{window_size}s/{model_name}_raw_scores_mw{mw2_target}.png")
        plt.show()

        
def plot_rf_feat_importance(importances, columns, window_size, classifier_type):
    """
    Unused in current pipeline.
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
    plt.savefig(f"./onset_v_self_report/{window_size}s/rf_feat_importance_{classifier_type}.png")
    plt.show()

def get_raw_scores(mw_models, sr_models, mw_2_models, X_mw, X_mw2, X_sr, mw2_target, 
                   full_set_relative_times, true_labels_full_set, window_size, feature_importance_flag):
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
    mw2_target: Float.
        The time (relative to the event) that training windows are centered on.
        Should be 2, 2.5, or 5.
    full_set_relative_times : Series
        Relative times for the full feature set, with indices matching X_mw, X_mw2, and X_sr
    true_labels_full_set : Series
        True labels for the full feature set, with indices matching X_mw, X_mw2, and X_sr.
        Values are as follows: 1 = mw_onset event, 2 = self_report event, 0 = control event.
    feature_importance_flag : Bool.
        Boolean flag specifying whether or not these raw scores are being calculated
        to later be used to calculate the forward model for finding feature importance.
        If True, features are filtered in the same way that training data is filtered
        for each model (by time and event).

    Returns
    -------
    results : Dict.
        Nested dictionary holding raw scores for each linear model (SVM, logistic 
        regression, LDA), for each model type (mw onset vs. control and self 
        report vs. control). First level keys are model names while second level 
        keys are mw_onset_raw_scores and self_report_raw_scores. 

    """
    # if feature impotance flag is true, calculate Z from a filtered subset of
    # full dataset that matches what the classifier was trained on 
    if feature_importance_flag: 
        # concat feature sets with metadata
        X_mw = pd.concat([X_mw, full_set_relative_times.rename("relative_time"), true_labels_full_set.rename("label")], axis=1)
        X_mw2 = pd.concat([X_mw2, full_set_relative_times.rename("relative_time"), true_labels_full_set.rename("label")], axis=1)
        X_sr = pd.concat([X_sr, full_set_relative_times.rename("relative_time"), true_labels_full_set.rename("label")], axis=1)

        # filter for mw
        X_mw = X_mw[
            ((X_mw["label"] == 0) & (X_mw["relative_time"] == 0)) |
            ((X_mw["label"] == 1) * (X_mw["relative_time"] == 0))
            ]
        # filter for mw2
        X_mw2 = X_mw2[
            ((X_mw2["label"] == 0) & (X_mw2["relative_time"] == mw2_target)) |
            ((X_mw2["label"] == 1) * (X_mw2["relative_time"] == mw2_target))
            ]
        # filter for sr
        X_sr = X_sr[
            ((X_sr["label"] == 0) & (X_sr["relative_time"] == (-.5 * window_size))) |
            ((X_sr["label"] == 2) * (X_sr["relative_time"] == (-.5 * window_size)))
            ]
    
        # drop metadata
        X_mw = X_mw.copy()
        X_mw.drop(columns=["label", "relative_time"], inplace=True)
        
        X_mw2 = X_mw2.copy()
        X_mw2.drop(columns=["label", "relative_time"], inplace=True)
        
        X_sr = X_sr.copy()
        X_sr.drop(columns=["label", "relative_time"], inplace=True)

    # these raw scores can later be used for calculating feature importance or
    # for plotting different event behavior as fntn of relative time
    #linear_models = ["Support Vector Machine", "Logistic Regression", "Linear Discriminant Analysis"]
    # focusing on logistic regression
    linear_models = ["Logistic Regression"]
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

    

def plot_forward_feature_importance(raw_scores, window_size, X_mw, X_mw2, X_sr,
                                    full_set_relative_times, true_labels_full_set,
                                    no_mw2_flag, mw2_target):
    """
    Unused in current pipeline.
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
    window_size : Int.
        The size of the sliding window for the data currently being processed.
    X_mw : DataFrame.
        Features for the full dataset, no PCA transform applied. The same scaler
        for the mw_onset vs. control classifier was applied to this.
    X_mw2 : DataFrame.
        Features for the full dataset, no PCA transform applied. The same scaler
        for the mw_onset_2 vs. control classifier was applied to this.  
    X_sr : DataFrame.
        Features for the full dataset, no PCA transform applied. The same scaler
        for the self_report vs. control classifier was applied to this.
    full_set_relative_times : Series
        Relative times for the full dataset, with index matching X.
    true_labels_full_set : Series
        True labels for the full dataset, with index matching X. 0 is control,
        1 is MW_onset, 2 is self_report.
    no_mw2_flag : Bool.
        Boolean flag representing whether or not to plot info related to the 
        mw_onset classifier trained on relative time != 0.
    mw2_target: Float.
        The time (relative to the event) that training windows are centered on.
        Should be 2, 2.5, or 5.


    Returns
    -------
    None.
    
    """
    
    # concat X with times and labels for filtering
    X_mw_combined = pd.concat([X_mw, full_set_relative_times.rename("relative_time"), true_labels_full_set.rename("label")], axis=1)
    if no_mw2_flag == False:
        X_mw2_combined = pd.concat([X_mw2, full_set_relative_times.rename("relative_time"), true_labels_full_set.rename("label")], axis=1)
    X_sr_combined = pd.concat([X_sr, full_set_relative_times.rename("relative_time"), true_labels_full_set.rename("label")], axis=1)

    # filter for mw
    X_mw = X_mw_combined[
        ((X_mw_combined["label"] == 0) & (X_mw_combined["relative_time"] == 0)) |
        ((X_mw_combined["label"] == 1) * (X_mw_combined["relative_time"] == 0))
        ]
    # filter for mw2
    if no_mw2_flag == False:
        X_mw2 = X_mw2_combined[
            ((X_mw2_combined["label"] == 0) & (X_mw2_combined["relative_time"] == mw2_target)) |
            ((X_mw2_combined["label"] == 1) * (X_mw2_combined["relative_time"] == mw2_target))
            ]
    # filter for sr
    X_sr = X_sr_combined[
        ((X_sr_combined["label"] == 0) & (X_sr_combined["relative_time"] == (-.5 * window_size))) |
        ((X_sr_combined["label"] == 2) * (X_sr_combined["relative_time"] == (-.5 * window_size)))
        ]
    
    
    # drop relative time and label from each filtered dataset
    X_mw = X_mw.copy()
    X_sr = X_sr.copy()
    X_mw.drop(columns = ["label", "relative_time"], inplace=True)
    X_sr.drop(columns = ["label", "relative_time"], inplace=True)
    if no_mw2_flag == False:
        X_mw2 = X_mw2.copy()
        X_mw2.drop(columns = ["label", "relative_time"], inplace=True)
        
    # get columns - they're the same for each set so choose any
    columns = X_mw.columns
    
    # calculate forward model feature importance and plot for each linear model, for each model type
    # one plot for each model feature importance with subplots for mw onset and sr model types
    
    
    for model_name, scores in raw_scores.items():
        Z_mw = scores["MW_onset_raw_scores"]
        Z_sr = scores["self_report_raw_scores"]
        if no_mw2_flag == False:
            Z_mw2 = scores["MW_onset_2_raw_scores"]
        
        # find A for mw model type
        A_numerator_mw = np.dot(X_mw.T, Z_mw) # transpose X so inner dimensions match for mult.
        A_denom_mw = np.dot(Z_mw.T, Z_mw)
        A_mw = A_numerator_mw / A_denom_mw # now A has one importance value for each feature
        
        if no_mw2_flag == False:
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
        if no_mw2_flag == False:
            importances_mw2 = np.abs(A_mw2)
    
        indices_mw = np.argsort(importances_mw)[::-1]  # Sort feature importances in descending order
        indices_sr = np.argsort(importances_sr)[::-1]  # Sort feature importances in descending order
        if no_mw2_flag == False:
            indices_mw2 = np.argsort(importances_mw2)[::-1] 
        
        if no_mw2_flag == False:
            fig, axes = plt.subplots(1,3,figsize=(20,10), sharey=True)
        else:
            fig, axes = plt.subplots(1,2, figsize=(20,10), sharey=True)
    
        # Plot feature importance
        # mw onset
        axes[0].bar(range(len(feature_names)), importances_mw[indices_mw])
        axes[0].set_xticks(range(len(feature_names)))
        axes[0].set_xticklabels([feature_names[i] for i in indices_mw], rotation=45, ha="right")
        axes[0].set_ylabel("Importance")
        axes[0].set_title("MW_Onset vs. Control")
        axes[0].grid(True)
        
        # self report
        axes[1].bar(range(len(feature_names)), importances_sr[indices_sr])
        axes[1].set_xticks(range(len(feature_names)))
        axes[1].set_xticklabels([feature_names[i] for i in indices_sr], rotation=45, ha="right")
        axes[1].set_ylabel("Importance")
        axes[1].set_title("Self-Report vs. Control")
        axes[1].grid(True)
        
        if no_mw2_flag == False:
            # mw onset 2
            axes[2].bar(range(len(feature_names)), importances_mw2[indices_mw2])
            axes[2].set_xticks(range(len(feature_names)))
            axes[2].set_xticklabels([feature_names[i] for i in indices_mw2], rotation=45, ha="right")
            axes[2].set_ylabel("Importance")
            axes[2].set_title(f"MW_Onset {mw2_target} vs. Control")
            axes[2].grid(True)
        
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle(f"Forward Model Feature Importance: {model_name} {window_size}s Sliding Window")
        if no_mw2_flag:
            plt.savefig(f"onset_v_self_report/{window_size}s/{model_name}_forward_feat_importance_no_mw2.png") 
        else:
            plt.savefig(f"onset_v_self_report/{window_size}s/{model_name}_forward_feat_importance_mw{mw2_target}.png") 
        plt.show()
        
def plot_forward_feature_importance_reformatted(raw_scores, window_size, X_mw, X_mw2, X_sr,
                                    full_set_relative_times, true_labels_full_set,
                                    no_mw2_flag, mw2_target, mw_color, sr_color, mw2_color):
    """
    Calculate and plot forward model feature importance for each linear model,
    for each model type (mw_onset vs. control or self report vs. control). Creates
    one plot for each linear model with two subplots: one for the mw_onset vs. control
    classifier and one for the self_report vs. control classifier.
    Instead of creating side by side subplots, this version creates a single bar chart.
    The bars are in order of the mw2 classifier if present, otherwise in order of the
    mw classifier, feature importance for all classifier types is represented 
    in this single figure.

    Parameters
    ----------
    raw_scores : Dict.
        Nested dictionary holding raw scores for each linear model (SVM, logistic 
        regression, LDA), for each model type (mw onset vs. control and self 
        report vs. control). First level keys are model names while second level 
        keys are mw_onset_raw_scores and self_report_raw_scores. 
    window_size : Int.
        The size of the sliding window for the data currently being processed.
    X_mw : DataFrame.
        Features for the full dataset, no PCA transform applied. The same scaler
        for the mw_onset vs. control classifier was applied to this.
    X_mw2 : DataFrame.
        Features for the full dataset, no PCA transform applied. The same scaler
        for the mw_onset_2 vs. control classifier was applied to this.  
    X_sr : DataFrame.
        Features for the full dataset, no PCA transform applied. The same scaler
        for the self_report vs. control classifier was applied to this.
    full_set_relative_times : Series
        Relative times for the full dataset, with index matching X.
    true_labels_full_set : Series
        True labels for the full dataset, with index matching X. 0 is control,
        1 is MW_onset, 2 is self_report.
    no_mw2_flag : Bool.
        Boolean flag representing whether or not to plot info related to the 
        mw_onset classifier trained on relative time != 0.
    mw_color : String, matplotlib CSS color name.
        The color to use for data related to mind wandering in this plot.
    sr_color : String, matplotlib CSS color name.
        The color to use for data related to self report in this plot.
    mw2_color : String, matplotlib CSS color name.
        The color to use for data related to the second mind wandering classifier
        in this plot. The second mind wandering classifier refers to the classifier
        trained on mw_onset events in windows centered at relative times later than time 0. 


    Returns
    -------
    None.
    
    """
    
    # concat X with times and labels for filtering
    X_mw_combined = pd.concat([X_mw, full_set_relative_times.rename("relative_time"), true_labels_full_set.rename("label")], axis=1)
    if no_mw2_flag == False:
        X_mw2_combined = pd.concat([X_mw2, full_set_relative_times.rename("relative_time"), true_labels_full_set.rename("label")], axis=1)
    X_sr_combined = pd.concat([X_sr, full_set_relative_times.rename("relative_time"), true_labels_full_set.rename("label")], axis=1)

    # filter for mw
    X_mw = X_mw_combined[
        ((X_mw_combined["label"] == 0) & (X_mw_combined["relative_time"] == 0)) |
        ((X_mw_combined["label"] == 1) * (X_mw_combined["relative_time"] == 0))
        ]
    # filter for mw2
    if no_mw2_flag == False:
        X_mw2 = X_mw2_combined[
            ((X_mw2_combined["label"] == 0) & (X_mw2_combined["relative_time"] == mw2_target)) |
            ((X_mw2_combined["label"] == 1) * (X_mw2_combined["relative_time"] == mw2_target))
            ]
    # filter for sr
    X_sr = X_sr_combined[
        ((X_sr_combined["label"] == 0) & (X_sr_combined["relative_time"] == (-.5 * window_size))) |
        ((X_sr_combined["label"] == 2) * (X_sr_combined["relative_time"] == (-.5 * window_size)))
        ]
    
    
    # drop relative time and label from each filtered dataset
    X_mw = X_mw.copy()
    X_sr = X_sr.copy()
    X_mw.drop(columns = ["label", "relative_time"], inplace=True)
    X_sr.drop(columns = ["label", "relative_time"], inplace=True)
    if no_mw2_flag == False:
        X_mw2 = X_mw2.copy()
        X_mw2.drop(columns = ["label", "relative_time"], inplace=True)
        
    # get columns - they're the same for each set so choose any
    columns = X_mw.columns
    
    # calculate forward model feature importance and plot for each linear model, for each model type
    # one plot for each model feature importance with subplots for mw onset and sr model types
    
    
    for model_name, scores in raw_scores.items():
        Z_mw = scores["MW_onset_raw_scores"]
        Z_sr = scores["self_report_raw_scores"]
        if no_mw2_flag == False:
            Z_mw2 = scores["MW_onset_2_raw_scores"]
        
        # find A for mw model type
        A_numerator_mw = np.dot(X_mw.T, Z_mw) # transpose X so inner dimensions match for mult.
        A_denom_mw = np.dot(Z_mw.T, Z_mw)
        A_mw = A_numerator_mw / A_denom_mw # now A has one importance value for each feature
        
        if no_mw2_flag == False:
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
        if no_mw2_flag == False:
            importances_mw2 = np.abs(A_mw2)
            
        
    
        #indices_mw = np.argsort(importances_mw)[::-1]  # Sort feature importances in descending order
        #indices_sr = np.argsort(importances_sr)[::-1]  # Sort feature importances in descending order
        if no_mw2_flag == False:
            # sort all feature importances so they are aligned with mw2 importances
            sort_indices = np.argsort(importances_mw2)[::-1] 
            sorted_features = [feature_names[i] for i in sort_indices]
            
            importances_mw = importances_mw[sort_indices]
            importances_mw2 = importances_mw2[sort_indices]
            importances_sr = importances_sr[sort_indices]
        else: # if no mw2, order by mw importances
            sort_indices = np.argsort(importances_mw)[::-1]
            sorted_features = [feature_names[i] for i in sort_indices]
            
            importances_mw = importances_mw[sort_indices]
            importances_sr = importances_sr[sort_indices]
    
        # set x axis to len feature names 
        x = np.arange(len(feature_names))
        bar_width = .25
        plt.figure(figsize=(20,15))
        ax = plt.gca()
        
        if no_mw2_flag == False: # include mw2, mw, and sr
            ax.bar(x - bar_width, importances_mw2, width = bar_width, label = f"MW_Onset_{mw2_target}", color = mw2_color)
            ax.bar(x, importances_mw, width = bar_width, label="MW_Onset", color= mw_color)
            ax.bar(x + bar_width, importances_sr, width = bar_width, label=f"self_report", color=sr_color)
        # plot when no mw2 is True
        else: # just include mw and sr
            ax.bar(x - bar_width, importances_mw, width = bar_width, label="MW_Onset", color=mw_color)
            ax.bar(x + bar_width, importances_sr, width = bar_width, label=f"self_report", color=sr_color)
            
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_features, rotation=45, ha="right")
        ax.set_ylabel("Importance (absolute value)")
        ax.set_title(f"Forward Model Feature Importance: {model_name} {window_size}s Sliding Window")
        ax.grid(True)
        ax.legend()
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if no_mw2_flag:
            plt.savefig(f"onset_v_self_report/{window_size}s/{model_name}_forward_feat_importance_no_mw2_new_layout.png") 
        else:
            plt.savefig(f"onset_v_self_report/{window_size}s/{model_name}_forward_feat_importance_mw{mw2_target}_new_layout.png") 
        plt.show()
        
def predictor_hist(raw_scores, true_labels_full_set, classifier_type, window_size, mw2_target, no_mw2_flag):
    """
    Unused in current pipeline.
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
    mw2_target : Int.
        The relative time that the second mw_onset vs. control classifier was 
        trained on. For example, 2 or 5. 
    no_mw2_flag: Bool.
        Boolean flag specifying whether or not to plot the classifier trained to 
        distinguish between mw_onset and control for the relative time NOT equal
        to 0.
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
        suptitle_addition = f"MW_onset {mw2_target} vs. Control"
        y = true_labels_full_set.apply(lambda x: 1 if x==1 else 0) # 1 if mw, else 0

    #fig, axes = plt.subplots(3,1,figsize=(15,15))
    # one fig now - only focusing on logistic regression
    fig, axes = plt.subplots(1,1, figsize=(15,15))
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
    if no_mw2_flag == False:
        plt.savefig(f"onset_v_self_report/{window_size}s/Predictor_hist_{classifier_type}_mw{mw2_target}.png")
    else: # dont specify mw2 target if no mw2
        plt.savefig(f"onset_v_self_report/{window_size}s/Predictor_hist_{classifier_type}.png")
        
def run_LOGOCV(X, y, groups, models, classifier_type, window_size, random_state,
               PCA_flag, PCA_thresh, smote_flag):
    """
    Input filtered X and y for the desired classifier type as well as the sub_id
    column for that subset. Completes LOGOCV for this classifier type with
    groups as subjects. 
    
    Although it appears that smote occurs outside of this pipeline, that's applied
    to the train sets while logocv operates on the full datasets (subsets for
    each classifier type) so it should be done again here.
    
    Accepts:
    X: DataFrame
        Full filtered feature set (no train test split) for the classifier of interest.
    y: Series
        Full filtered labels set (no train test split) for the classifier of interest.
    groups: Series
        LOGOCV groups (subject ids) for this classifier after filtering 
        has been applied (still no train test split prior, though)
    models: Dict.
        Dictionary of initialized models for cross validation for this classifier type.
        First level keys are model names, ex. Logistic Regression.
    classifier_type: String.
        The classifier type being trained ex. Self Report or MW Onset.
    window_size: Int
        The size of the windows for the data being used in this pipeline. 2 or 5.
    random_state: Int
        The random state used throughout this pipeline.
    PCA_flag: Bool
        A flag indicating whether or not to use PCA for LOGOCV with this classifier
        type.
    PCA_thresh: Float
        The amount of variance to retain if PCA is used. Specify as a decimal 
        ex. .95 to retain 95% of variance.
    smote_flag: Bool
        A flag indicating whether or not to use SMOTE to generate synthetic data
        and balance class distribution. 
    """
    # new logocv object for this classifier type
    logo = LeaveOneGroupOut()
    # initialize lists to store results
    foldwise_results = []
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups = groups)):
        print(f" --- Fold {fold+1} ({classifier_type}) --- ")
        X_train_fold, X_test_fold = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train_fold, y_test_fold = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        
        # get test subject
        test_subject = groups.iloc[test_idx].unique()[0] # only one per group
        
        if smote_flag:
            smote = SMOTE(random_state = random_state)
            X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)
            
        # train - scaler applied within function
        models, results = train_models_cv(X_train_fold, y_train_fold, X_test_fold, y_test_fold, 
                                          models, PCA_flag, random_state, PCA_thresh)
        
        # add test sub to results
        results["test_subject"] = test_subject
        
        foldwise_results.append(results)
        
    return foldwise_results

def plot_roc(foldwise_results, classifier_type, window_size, model_names,
             mw_color, sr_color, mw2_color):
    """
    Plot the ROC curve by concatenating AUROC over folds from cross validation.
    Save the plot.
    
    Accepts:
    -------
    foldwise_results: List.
        List of dictionaries, there is one listitem/ dictionary for each LOGOCV
        fold. Each dictionary has two keys, the model name (usually Logistic Regression)
        and "test_subject". The model name key holds another dictionary of size two,
        where keys are "true_labels" and "y_scores". As the names imply, these
        hold true labels and y_scores for the LOGOCV fold represented by this
        top-level list entry. "test_subject" holds the subject id for the subject
        that the classifiers were tested on for this fold (since we are doing 
        LOGOCV over subjects, there's one "test subject" for each fold).
    classifier_type: String.
        A string representing the type of classfier that we are plotting the 
        ROC curve for. This must be "Self Report", "MW Onset", "MW Onset 2.5", 
        "MW Onset 2", or "MW Onset 5". It is important to reference the event 
        type targeted by the classifier and the relative time that training 
        windows are centered on (when not time 0) so that it is clear what the 
        plot is for as this is used in the plot title. The string must be one of 
        the listed values in this description for color assignment to work properly.
    window_size: Int.
        The size of the window being used. Usually 2 or 5. Used for the save path.
    model_names: List.
        A list of strings representing the models that we are plotting ROC curves
        for. Typically this will just be Logistic Regression, but we can add to this
        to plot multiple ROC curves at once if training multiple model types.
    mw_color : String, matplotlib CSS color name.
        The color to use for data related to mind wandering in this plot.
    sr_color : String, matplotlib CSS color name.
        The color to use for data related to self report in this plot.
    mw2_color : String, matplotlib CSS color name.
        The color to use for data related to the second mind wandering classifier
        in this plot. The second mind wandering classifier refers to the classifier
        trained on mw_onset events in windows centered at relative times later than time 0. 
    
    Returns
    -------
    None.
        
    """
    # Plot ROC Curve
    # code adapted from HS
    
    # set color based on classifier type
    if classifier_type == "Self Report":
        color = sr_color
    elif classifier_type == "MW Onset":
        color = mw_color
    elif (classifier_type == "MW Onset 2") | (classifier_type == "MW Onset 2.5") | (classifier_type == "MW Onset 5"):
        color = mw2_color
    
    # flatten foldwise results dictionary and concat to feed into roc curve function
    
    plt.figure(figsize = (12,12))
    
    for model_name in model_names: # loop through model names
        all_y_scores = []
        all_true_labels = []
        
        for fold in foldwise_results: # loop through folds to get y scores and y true
            all_y_scores.extend(fold[model_name]["y_scores"])
            all_true_labels.extend(fold[model_name]["true_labels"])
            
        fpr, tpr, _ = roc_curve(all_true_labels, all_y_scores)
        plt.plot(fpr, tpr, label=f"{model_name} AUROC = {roc_auc_score(all_true_labels, all_y_scores):.2f})", color=color)
        
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(f"LOGOCV ROC Curve: {classifier_type}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"onset_v_self_report/{window_size}s/ROC_curve_{classifier_type}.png")
    
def plot_foldwise_auroc(foldwise_results_mw, foldwise_results_sr, window_size,
                        model_names, mw_color, sr_color, mw2_color,
                        mw2_target= None, foldwise_results_mw2 = None):
    """
    Plots a bar chart of auroc for each fold of LOGOCV. Labels denoting the subject id that
    the model was tested on are represented on the x-axis. All classifier types
    are represented in one figure and bars are positioned in descending order
    of auroc for the mw2 classifier if present. If not present, bars are positioned
    in descending order of auroc for the mw classifier.
    
    References get_p_values to get p-values for each fold, for each classifier and
    model type. Asterisks are placed above the bars for folds where the p-value
    is less than 0.05, representing that these scores are statistically significantly
    better than a random classifier (AUROC of .5).
    
    Accepts:
    -------
    foldwise_resullts_mw : List.
        List of dictionaries, there is one listitem/ dictionary for each LOGOCV
        fold during LOGOCV for the mind wandering classifier. Each dictionary 
        has two keys, the model name (usually Logistic Regression)
        and "test_subject". The model name key holds another dictionary of size two,
        where keys are "true_labels" and "y_scores". As the names imply, these
        hold true labels and y_scores for the LOGOCV fold represented by this
        top-level list entry. "test_subject" holds the subject id for the subject
        that the classifiers were tested on for this fold (since we are doing 
        LOGOCV over subjects, there's one "test subject" for each fold).
    foldwise_results_sr: List
        The same list of dictionaries as seen in foldwise_results_mw, but holding
        results from LOGOCV for the self report classifier.
    window_size: Int
        The size of the window used for data. 2 or 5. Used in plot title and save 
        path.
    model_names: List.
        A list of strings representing the models that we are plotting ROC curves
        for. Typically this will just be Logistic Regression, but we can add to this
        to plot multiple ROC curves at once if training multiple model types.
    mw_color : String, matplotlib CSS color name.
        The color to use for data related to mind wandering in this plot.
    sr_color : String, matplotlib CSS color name.
        The color to use for data related to self report in this plot.
    mw2_color : String, matplotlib CSS color name.
        The color to use for data related to the second mind wandering classifier
        in this plot. The second mind wandering classifier refers to the classifier
        trained on mw_onset events in windows centered at relative times later than time 0. 
    mw2_target: Float.
        The center of the windows used for training data for the mw2 classifier if present.
    foldwise_results_mw2: List.
        The same list of dictionaries as seen in foldwise_results_mw and foldwise_results_sr,
        but holding results from LOGOCV for the mw2 classifier.
        
    Returns
    -------
    None.
    
    """
    all_results = {}
    
    # conditionally add mw2 to all results - must be first so its ordered first
    mw2_label = f"MW Onset {mw2_target}"
    if foldwise_results_mw2 is not None:
        all_results[mw2_label] = foldwise_results_mw2
        # get p_values for mw2 if present
        print("Performing foldwise permutation tests for MW2")
        mw2_p_values = get_p_values(foldwise_results_mw2)
        
    # now add the other two and get their p values
    all_results["MW Onset"] = foldwise_results_mw
    print("Performing foldwise permutation tests for MW")
    mw_p_values = get_p_values(foldwise_results_mw)
    print("Performing foldwise permutation tests for SR")
    all_results["Self Report"] = foldwise_results_sr
    sr_p_values = get_p_values(foldwise_results_sr)


    for model_name in model_names: # new figure for each model (realistically there will probably only be one- logistic regression)
        if model_name == "test_subject":
            continue # skip this key if its not a model name
        classifier_data = {key: {"test_subjects": [], "aurocs": [], "p_values": []} for key in all_results}
    
        for classifier_type, results in all_results.items():
            for fold_idx, fold in enumerate(results):
                # get data
                y_scores = fold[model_name]["y_scores"]
                true_labels = fold[model_name]["true_labels"]
                test_subject = fold["test_subject"]
                # get p value
                if classifier_type == "MW Onset":
                    p_value = mw_p_values[model_name][fold_idx]
                elif classifier_type == "Self Report":
                    p_value = sr_p_values[model_name][fold_idx]
                elif (classifier_type == "MW Onset 2") | (classifier_type == "MW Onset 2.5") | (classifier_type == "MW Onset 5"):
                    p_value = mw2_p_values[model_name][fold_idx]
                
                # calculate auroc
                unique_vals = np.unique(true_labels)

                if len(unique_vals) < 2:
                    auroc = np.nan
                else:
                    auroc = roc_auc_score(true_labels, y_scores)
                
                # append to lists
                classifier_data[classifier_type]["test_subjects"].append(test_subject)
                classifier_data[classifier_type]["aurocs"].append(auroc)
                classifier_data[classifier_type]["p_values"].append(p_value)
            
        # determine sorting order
        if mw2_label in classifier_data:
            sort_by = mw2_label
        else:
            sort_by = "MW Onset"
        
        # sort auroc, subject in order of descending auroc for reference classifier
        auroc_sub_p_pairs = list(zip(classifier_data[sort_by]["test_subjects"], classifier_data[sort_by]["aurocs"], classifier_data[sort_by]["p_values"])) # zip together
        # handle nans when sorting
        sorted_auroc_sub_p_pairs = sorted(auroc_sub_p_pairs, key=lambda x: (np.isnan(x[1]),
                                                                        -x[1] if not np.isnan(x[1]) else float('-inf')))
        sort_by_subs, sort_by_aurocs, sort_by_p_values = zip(*sorted_auroc_sub_p_pairs)# separate the lists again
        
        x = np.arange(len(sort_by_subs))
        bar_width = .25
        
        # plot the figure for this model
        plt.figure(figsize=(20,16))
        
        # plot data for all classifier types
        for idx, (classifier_type, data) in enumerate(classifier_data.items()):
            # set color based on classifier type
            if classifier_type == "MW Onset":
                color = mw_color
            elif classifier_type == "Self Report":
                color = sr_color
            elif (classifier_type == "MW Onset 2") | (classifier_type == "MW Onset 2.5") | (classifier_type == "MW Onset 5"):
                color = mw2_color
            # map subs to auroc for this classifier type
            subject_to_auroc = dict(zip(data["test_subjects"], data["aurocs"]))
            # map subs to p values for this classifier type
            subject_to_p = dict(zip(data["test_subjects"], data["p_values"]))
            # sort aurocs according to reference classifier
            sorted_aurocs = [subject_to_auroc.get(sub, np.nan) for sub in sort_by_subs]
            sorted_ps = [subject_to_p.get(sub, np.nan) for sub in sort_by_subs]
            
            plt.bar(x+ (idx * bar_width), sorted_aurocs, width = bar_width, label=classifier_type, color=color)
            plt.axhline(y=0.5, color="black", linestyle='--', linewidth=1)
            
            # annotate bar with * when statistically significant (p < .05) - above the bar
            # same color as bar (dependent on classifier type)
            for i, (auroc, p_val) in enumerate(zip(sorted_aurocs, sorted_ps)):
                if not np.isnan(auroc) and p_val < 0.05:
                    plt.text(x[i] + (idx * bar_width), auroc + .01, "*",
                             ha="center", va="bottom", color = color)
            
        plt.xticks(x+ bar_width * len(classifier_data), sort_by_subs, rotation=45, ha="right")
        plt.xlabel("Test Subject")
        plt.ylabel("AUROC")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.grid(True)
        plt.legend()
        plt.title(f"Foldwise AUROC: LOGOCV Over Subjects: {model_name}, {window_size}s Sliding Window")
        plt.figtext(0.5, 0.0001, "* means p_value < 0.05 (AUROC is significantly greater than chance)", 
            ha="center", va="bottom")
        plt.savefig(f"onset_v_self_report/{window_size}s/foldwise_auroc_{model_name}.png")
        plt.close()
        
def plot_foldwise_auroc_subplots(foldwise_results_mw, foldwise_results_sr, window_size,
                        model_names, mw_color, sr_color, mw2_color,
                        mw2_target= None, foldwise_results_mw2 = None):
    """
    Plots a bar chart of auroc for each fold of LOGOCV. AUROC for each classifier
    is displayed within a separate subplot, but all bars are positioned in descending order
    of auroc for the mw2 classifier if present. If not present, bars are positioned
    in descending order of auroc for the mw classifier. Labels denoting the subject 
    id that the model was tested on are represented on the x-axis. 
    References get_p_values to get p-values for each fold, for each classifier and
    model type. Asterisks are placed above the bars for folds where the p-value
    is less than 0.05, representing that these scores are statistically significantly
    better than a random classifier (AUROC of .5).
    
    Accepts:
    -------
    foldwise_resullts_mw : List.
        List of dictionaries, there is one listitem/ dictionary for each LOGOCV
        fold during LOGOCV for the mind wandering classifier. Each dictionary 
        has two keys, the model name (usually Logistic Regression)
        and "test_subject". The model name key holds another dictionary of size two,
        where keys are "true_labels" and "y_scores". As the names imply, these
        hold true labels and y_scores for the LOGOCV fold represented by this
        top-level list entry. "test_subject" holds the subject id for the subject
        that the classifiers were tested on for this fold (since we are doing 
        LOGOCV over subjects, there's one "test subject" for each fold).
    foldwise_results_sr: List
        The same list of dictionaries as seen in foldwise_results_mw, but holding
        results from LOGOCV for the self report classifier.
    window_size: Int
        The size of the window used for data. 2 or 5. Used in plot title and save 
        path.
    model_names: List.
        A list of strings representing the models that we are plotting ROC curves
        for. Typically this will just be Logistic Regression, but we can add to this
        to plot multiple ROC curves at once if training multiple model types.
    mw_color : String, matplotlib CSS color name.
        The color to use for data related to mind wandering in this plot.
    sr_color : String, matplotlib CSS color name.
        The color to use for data related to self report in this plot.
    mw2_color : String, matplotlib CSS color name.
        The color to use for data related to the second mind wandering classifier
        in this plot. The second mind wandering classifier refers to the classifier
        trained on mw_onset events in windows centered at relative times later than time 0. 
    mw2_target: Float.
        The center of the windows used for training data for the mw2 classifier if present.
    foldwise_results_mw2: List.
        The same list of dictionaries as seen in foldwise_results_mw and foldwise_results_sr,
        but holding results from LOGOCV for the mw2 classifier.
        
    Returns
    -------
    None.
    
    """
    all_results = {}
    
    # conditionally add mw2 to all results - must be first so its ordered first
    mw2_label = f"MW Onset {mw2_target}"
    if foldwise_results_mw2 is not None:
        all_results[mw2_label] = foldwise_results_mw2
        # get p_values for mw2 if present
        print("Performing foldwise permutation tests for MW2")
        mw2_p_values = get_p_values(foldwise_results_mw2)
        
    # now add the other two and get their p values
    all_results["MW Onset"] = foldwise_results_mw
    print("Performing foldwise permutation tests for MW")
    mw_p_values = get_p_values(foldwise_results_mw)
    print("Performing foldwise permutation tests for SR")
    all_results["Self Report"] = foldwise_results_sr
    sr_p_values = get_p_values(foldwise_results_sr)


    for model_name in model_names: # new figure for each model (realistically there will probably only be one- logistic regression)
        if model_name == "test_subject":
            continue # skip this key if its not a model name
        classifier_data = {key: {"test_subjects": [], "aurocs": [], "p_values": []} for key in all_results}
    
        for classifier_type, results in all_results.items():
            for fold_idx, fold in enumerate(results):
                # get data
                y_scores = fold[model_name]["y_scores"]
                true_labels = fold[model_name]["true_labels"]
                test_subject = fold["test_subject"]
                # get p value
                if classifier_type == "MW Onset":
                    p_value = mw_p_values[model_name][fold_idx]
                elif classifier_type == "Self Report":
                    p_value = sr_p_values[model_name][fold_idx]
                elif (classifier_type == "MW Onset 2") | (classifier_type == "MW Onset 2.5") | (classifier_type == "MW Onset 5"):
                    p_value = mw2_p_values[model_name][fold_idx]
                
                # calculate auroc
                unique_vals = np.unique(true_labels)

                if len(unique_vals) < 2:
                    auroc = np.nan
                else:
                    auroc = roc_auc_score(true_labels, y_scores)
                
                # append to lists
                classifier_data[classifier_type]["test_subjects"].append(test_subject)
                classifier_data[classifier_type]["aurocs"].append(auroc)
                classifier_data[classifier_type]["p_values"].append(p_value)
            
        # determine sorting order
        if mw2_label in classifier_data:
            sort_by = mw2_label
        else:
            sort_by = "MW Onset"
        
        # sort auroc, subject in order of descending auroc for reference classifier
        auroc_sub_p_pairs = list(zip(classifier_data[sort_by]["test_subjects"], classifier_data[sort_by]["aurocs"], classifier_data[sort_by]["p_values"])) # zip together
        # handle nans when sorting
        sorted_auroc_sub_p_pairs = sorted(auroc_sub_p_pairs, key=lambda x: (np.isnan(x[1]),
                                                                        -x[1] if not np.isnan(x[1]) else float('-inf')))
        sort_by_subs, sort_by_aurocs, sort_by_p_values = zip(*sorted_auroc_sub_p_pairs)# separate the lists again
        
        # find min auroc across all classifiers to set ylim min
        all_aurocs = []
        for data in classifier_data.values():
            all_aurocs.extend(data["aurocs"])
        # handle na
        all_aurocs = [auroc for auroc in all_aurocs if not np.isnan(auroc)]
        y_min = min(all_aurocs)
        
        # get classifier types, initalize figure/ subplots for this model
        labels = list(classifier_data.keys()) # classifier types as labels
        n = len(labels)
        
        fig, axes = plt.subplots(1,n, figsize=(10*n, 20), sharey=True)
        if n == 1:
            axes = [axes]
        
        # plot data for all classifier types
        for ax, classifier_type in zip(axes, labels):
            data = classifier_data[classifier_type]

            # map subs to auroc for this classifier type
            subject_to_auroc = dict(zip(data["test_subjects"], data["aurocs"]))
            # map subs to p values for this classifier type
            subject_to_p = dict(zip(data["test_subjects"], data["p_values"]))
            # sort aurocs according to reference classifier
            sorted_aurocs = [subject_to_auroc.get(sub, np.nan) for sub in sort_by_subs]
            sorted_ps = [subject_to_p.get(sub, np.nan) for sub in sort_by_subs]
            # define x axis
            x = np.arange(len(sort_by_subs))
            
            # set color based on classifier type
            if classifier_type == "MW Onset":
                color = mw_color
            elif classifier_type == "Self Report":
                color = sr_color
            elif (classifier_type == "MW Onset 2") | (classifier_type == "MW Onset 2.5") | (classifier_type == "MW Onset 5"):
                color = mw2_color

                    
            # plot for this classifier
            ax.bar(x, sorted_aurocs, color=color)
            ax.axhline(0.5, color="black", linestyle='--', linewidth=1)
            
            # set y lim to reduce redundant info based on min auroc across all classifiers
            ax.set_ylim(y_min, 1.05) # leave some extra space for asterisks when auroc is 1
            # add asterisks for statistical significance
            for i, (auroc, p_val) in enumerate(zip(sorted_aurocs, sorted_ps)):
                if not np.isnan(auroc) and p_val < 0.05:
                    ax.text(i, auroc + .01, "*", ha="center", va="bottom", color = color)
            # ax level plot info
            ax.set_xticks(x)
            ax.set_xticklabels(sort_by_subs, rotation=90, ha="center", va="top")
            ax.set_title(classifier_type)
            ax.set_xlabel("Test Subject")
            ax.grid(True)
                
        axes[0].set_ylabel("AUROC")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle(f"Foldwise AUROC: LOGOCV Over Subjects: {model_name}, {window_size}s Sliding Window")
        plt.figtext(0.5, 0.0001, "* means p_value < 0.05 (AUROC is significantly greater than chance)", 
            ha="center", va="bottom")
        plt.savefig(f"onset_v_self_report/{window_size}s/foldwise_auroc_{model_name}_subplots.png")
        plt.close()
        
def plot_foldwise_auroc_histogram(foldwise_results_mw, foldwise_results_sr, window_size,
                        model_names, mw_color, sr_color, mw2_color,
                        mw2_target= None, foldwise_results_mw2 = None):
    """
    Plots a histogram of auroc for each fold of LOGOCV, representing how many
    scores were significantly above chance as determined through permutation
    tests.

    References get_p_values to get p-values for each fold, for each classifier and
    model type. 
    
    Accepts:
    -------
    foldwise_resullts_mw : List.
        List of dictionaries, there is one listitem/ dictionary for each LOGOCV
        fold during LOGOCV for the mind wandering classifier. Each dictionary 
        has two keys, the model name (usually Logistic Regression)
        and "test_subject". The model name key holds another dictionary of size two,
        where keys are "true_labels" and "y_scores". As the names imply, these
        hold true labels and y_scores for the LOGOCV fold represented by this
        top-level list entry. "test_subject" holds the subject id for the subject
        that the classifiers were tested on for this fold (since we are doing 
        LOGOCV over subjects, there's one "test subject" for each fold).
    foldwise_results_sr: List
        The same list of dictionaries as seen in foldwise_results_mw, but holding
        results from LOGOCV for the self report classifier.
    window_size: Int
        The size of the window used for data. 2 or 5. Used in plot title and save 
        path.
    model_names: List.
        A list of strings representing the models that we are plotting ROC curves
        for. Typically this will just be Logistic Regression, but we can add to this
        to plot multiple ROC curves at once if training multiple model types.
    mw_color : String, matplotlib CSS color name.
        The color to use for data related to mind wandering in this plot.
    sr_color : String, matplotlib CSS color name.
        The color to use for data related to self report in this plot.
    mw2_color : String, matplotlib CSS color name.
        The color to use for data related to the second mind wandering classifier
        in this plot. The second mind wandering classifier refers to the classifier
        trained on mw_onset events in windows centered at relative times later than time 0. 
    mw2_target: Float.
        The center of the windows used for training data for the mw2 classifier if present.
    foldwise_results_mw2: List.
        The same list of dictionaries as seen in foldwise_results_mw and foldwise_results_sr,
        but holding results from LOGOCV for the mw2 classifier.
        
    Returns
    -------
    None.
    
    """
    all_results = {}
    
    # conditionally add mw2 to all results - must be first so its ordered first
    mw2_label = f"MW Onset {mw2_target}"
    if foldwise_results_mw2 is not None:
        all_results[mw2_label] = foldwise_results_mw2
        # get p_values for mw2 if present
        print("Performing foldwise permutation tests for MW2")
        mw2_p_values = get_p_values(foldwise_results_mw2)
        
    # now add the other two and get their p values
    all_results["MW Onset"] = foldwise_results_mw
    print("Performing foldwise permutation tests for MW")
    mw_p_values = get_p_values(foldwise_results_mw)
    print("Performing foldwise permutation tests for SR")
    all_results["Self Report"] = foldwise_results_sr
    sr_p_values = get_p_values(foldwise_results_sr)

    # get all aurocs and p values (for each fold)
    for model_name in model_names: # new figure for each model (realistically there will probably only be one- logistic regression)
        if model_name == "test_subject":
            continue # skip this key if its not a model name
        classifier_data = {}
    
        for classifier_type, results in all_results.items():
            aurocs = []
            p_values = []
            # process each fold
            for result_idx, fold in enumerate(results):
                y_scores = fold[model_name]["y_scores"]
                true_labels = fold[model_name]["true_labels"]
                
                # grab the p value for this result (dependent on model name, idx, and classifier type)
                if classifier_type == "MW Onset":
                    p_value = mw_p_values[model_name][result_idx]
                elif classifier_type == "Self Report":
                    p_value = sr_p_values[model_name][result_idx]
                elif classifier_type == f"MW Onset {mw2_target}":
                    p_value = mw2_p_values[model_name][result_idx]
                    
                # compute auroc from true labels and y scores
                # nan if only one result type
                if len(np.unique(true_labels)) < 2:
                    auroc = np.nan
                else:
                    auroc = roc_auc_score(true_labels, y_scores)
                    
                # add to lists
                aurocs.append(auroc)
                p_values.append(p_value)
            # add this folds data to the dictionary under this classifiers key
            classifier_data[classifier_type] = {
                "aurocs" : np.array(aurocs),
                "p_values": np.array(p_values)
                }
        
        # plot - subplot for each classifier type
        num_classifiers = len(classifier_data)
        fig, axes = plt.subplots(1, num_classifiers, figsize=(20,15), sharey=True)
        
        if num_classifiers == 1:
            axes = [axes] # handle when theres only one classifier
        
        for ax, (classifier, data) in zip(axes, classifier_data.items()):
            # get auroc and p val
            aurocs = data["aurocs"]
            p_vals = data["p_values"]
            
            # set valid flag to filter out auroc and p val when auroc is na
            valid = ~np.isnan(aurocs)
            aurocs = aurocs[valid]
            p_vals = p_vals[valid]
            
            # sort aurocs to determine significance threshold
            sorted_idx = np.argsort(aurocs)
            sorted_aurocs = aurocs[sorted_idx]
            sorted_ps = p_vals[sorted_idx]
            threshold = None
            # set threshold to be at the auroc for which all higher aurocs are 
            # statistically significant
            for i in range(len(sorted_aurocs)):
                if np.all(sorted_ps[i:] < 0.05):
                    threshold = sorted_aurocs[i]
                    break
            # set color depending on classifier type
            if classifier == "MW Onset":
                color = mw_color
            elif classifier == "Self Report":
                color = sr_color
            elif classifier == f"MW Onset {mw2_target}":
                color = mw2_color
            
            # plot
            bins = np.linspace(min(aurocs), max(aurocs), 11) # 11 bins ranging from min to max auroc
            ax.hist(aurocs, bins = bins, color = color)
            ax.set_title(classifier)
            ax.set_xlabel("AUROC")
            ax.set_ylabel("Number of Test Subjects")
            ax.grid(True)
            
            if threshold is not None:
                ax.axvline(threshold, color="black", linestyle="--", label = f"All AUROCs >= {threshold: .2f} \nare stastically significantly\n greater than chance")
                ax.legend(fontsize=10)
                
        plt.suptitle(f"AUROC Histograms with Stastical Significance Cutoff. Trained via LOGOCV. {model_name}, {window_size}s Window")
        plt.tight_layout()
        plt.savefig(f"onset_v_self_report/{window_size}s/foldwise_auroc_{model_name}_histogram.png")
        plt.close()

    
    
def get_p_values(foldwise_results, num_permutations = 1000):
    """
    Performs a permutation test to assess whether foldwise AUROC scores 
    are statistically significantly greater than chance (AUROC = 0.5).
    
    Tests the null hypothesis that the model's performance for each fold is 
    no better than random classification against the alternative hypothesis
    that the model's performance for each fold is greater than .5.

    Parameters
    ----------
    foldwise_resullts : List.
        List of dictionaries, there is one listitem/ dictionary for each LOGOCV
        fold during LOGOCV for one classifier. Each dictionary 
        has two keys, the model name (usually Logistic Regression)
        and "test_subject". The model name key holds another dictionary of size two,
        where keys are "true_labels" and "y_scores". As the names imply, these
        hold true labels and y_scores for the LOGOCV fold represented by this
        top-level list entry. "test_subject" holds the subject id for the subject
        that the classifiers were tested on for this fold (since we are doing 
        LOGOCV over subjects, there's one "test subject" for each fold).
    num_permutations: Int.
        The number of permutations to perform.

    Returns
    -------
    p_values: Dict.
        Dictionary holding p-values for each model, for each fold. First level
        keys are model names (usually there will just be one, Logistic Regression).
        These keys hold lists where the value for each list entry is a p-value
        indicating the significance of the observed AUROC scores for each fold. 
        Listitems are in order of folds such that the p-value at 
        p_values["model_name"][0] corresponds to LOGOCV fold 0.

    """
    # initialize dict to store p-values for each model, for each fold
    p_values = {model_name: [] for model_name in model_names}
    
    # for each fold
    for i, fold in enumerate(foldwise_results):
        print(f"Fold {i}/{len(foldwise_results)}")
        # for each model for this fold
        for model_name in fold:
            if model_name == "test_subject":
                continue # skip the rest of the loop if this is just the test sub keu
            # get actual auroc using true labels and y scores for this fold
            true_labels = fold[model_name]["true_labels"]
            y_scores = fold[model_name]["y_scores"]
            actual_auroc = roc_auc_score(true_labels, y_scores)
            
            # initialize list to store aurocs from null dist.
            null_aurocs = []
            # generate null dist. via permutations
            for permutation in range(num_permutations +1): 
                if permutation % 100 == 0:
                    print(f"Permutation {permutation}/{num_permutations}")
                # shuffle true labels, keeping predictions fixed
                shuffled_labels = np.random.permutation(true_labels)
                # recalc. AUROC with shuffled labels
                null_auroc = roc_auc_score(shuffled_labels, y_scores)
                null_aurocs.append(null_auroc)

            # calculate p-value (one sided b/c we want to see if AUROC significantly > .5, not just != .5)
            p_value = np.mean(np.array(null_aurocs) >= actual_auroc)
            # save to p_values list under model name key in dict
            p_values[model_name].append(p_value)
            
    return p_values
    
    
#%%
    

window_size = 5 # hyperparams/ pca/ smote have not been optimized for the 2s window
model_names = ["Logistic Regression"] # list the models you are working with as denoted in the results dictionaries

# define the colors you'd like to use in plots (matplotlib color names: https://matplotlib.org/stable/gallery/color/named_colors.html)
mw_color =  "cadetblue" # the color for mind wandering in plots
sr_color =  "seagreen" # the color for self report in plots
mw2_color =  "orangered" # the color for mind wandering 2 in plots (this is the second mw_onset classifier, can be trained on windows centered at various times ex. 2.5, 5)

PCA_flag_self_report = False
PCA_threshold_self_report = None
PCA_flag_MW_onset = False
PCA_threshold_MW_onset = None
PCA_flag_MW_onset_2_5 = False 
PCA_threshold_MW_onset_2_5 = None
PCA_flag_MW_onset_2 = True
PCA_threshold_MW_onset_2 = .95
PCA_flag_MW_onset_5 = True
PCA_threshold_MW_onset_5 = .95

random_state = 42 # shouldn't need to change this
SMOTE_flag = True # set to True to use SMOTE (synthetic minority over-sampling technique) to achieve more balanced class dist.
undersample_flag = False # set to True to use undersampling to achieve more balanced class dist. 
#PCA_flag = True
#PCA_threshold = .9 # for PCA
# optimal PCA_flag and threshold is model specific
no_mw2_flag = False # set to true to prevent plotting of the MW_onset vs control classifier trained on relative time != 0
mw2_target = 2.5 # set to target time for the second mw_onset vs. control classifier. currently 2, 2.5, and 5 are supported.

# DO NOT SET BOTH UNDERSAMPLE_FLAG AND SMOTE_FLAG TO TRUE AT THE SAME TIME THAT WOULD BE WEIRD/ ERROR WILL BE THROWN
assert not (SMOTE_flag == True and undersample_flag == True), "Error: cannot use SMOTE and undersampling at the same time this way. Change one to False or adapt the pipeline."
    

# load data

# to do: load in train and test
X_train = pd.read_csv(f"X_train_wlen{window_size}.csv", index_col=0)
y_train = pd.read_csv(f"y_train_wlen{window_size}.csv", index_col=0).squeeze("columns")
X_test = pd.read_csv(f"X_test_wlen{window_size}.csv", index_col=0)
y_test = pd.read_csv(f"y_test_wlen{window_size}.csv", index_col=0).squeeze("columns")
X = pd.read_csv(f"X_wlen{window_size}.csv", index_col = 0)
y = pd.read_csv(f"y_wlen{window_size}.csv", index_col = 0).squeeze("columns")
trial_count_df = pd.read_csv(f"trial_counts_wlen{window_size}.csv")


# from train set, filter out the following - leave the test set untouched
# rows where label = control and relative_time != 0
# rows where label = self report and relative_time != -1 if window_size = 2, -2.5 if window_size = 5
# rows where label = mw_onset and relative_time != 0

X_train = X_train[
    ((X_train["label"] == "control") & (X_train["relative_time"] == 0)) |
    ((X_train["label"] == "self_report") & (X_train["relative_time"] == (-.5 * window_size))) |
    ((X_train["label"] == "MW_onset") & (X_train["relative_time"] == 0)) |
    ((X_train["label"] == "MW_onset") & (X_train["relative_time"] == mw2_target)) | # add condition for mw2_target
    ((X_train["label"] == "control") & (X_train["relative_time"] == mw2_target)) |
    ((X_train["label"] == "control") & (X_train["relative_time"] == (-.5 * window_size)))] # retain some control times at rel time mw2 target and -.5* window as well

# apply same filter logic to a filtered x test for alternative performance metric calculations
X_test_filtered = X_test[
    ((X_test["label"] == "control") & (X_test["relative_time"] == 0)) |
    ((X_test["label"] == "self_report") & (X_test["relative_time"] == (-.5 * window_size))) |
    ((X_test["label"] == "MW_onset") & (X_test["relative_time"] == 0)) |
    ((X_test["label"] == "MW_onset") & (X_test["relative_time"] == mw2_target)) | 
    ((X_test["label"] == "control") & (X_test["relative_time"] == mw2_target)) |
    ((X_test["label"] == "control") & (X_test["relative_time"] == (-.5 * window_size)))
    ]

# filter x to support classifier specific x subsets for logocv
X_filtered = X[
    ((X["label"] == "control") & (X["relative_time"] ==0)) |
    ((X["label"] == "self_report") & (X["relative_time"] == (-.5 * window_size))) |
    ((X["label"] == "MW_onset") & (X["relative_time"] == 0)) |
    ((X["label"] == "MW_onset") & (X["relative_time"] == mw2_target)) | # add condition for mw2_target
    ((X["label"] == "control") & (X["relative_time"] == mw2_target)) |
    ((X["label"] == "control") & (X["relative_time"] == (-.5 * window_size)))]

# sanity checks.. should have all three label types and only 0 as relative time 
# val for each group except self_report, which should have only -1 or -2.5 depending on window size.
print("unique vals for label column:", X_train["label"].unique())
print("unique vals for relative time, groupedby label: ")
print(X_train.groupby("label")["relative_time"].unique())

# adjust y_train accordingly so that it matches full X_train (same rows dropped)
y_train = y_train.loc[X_train.index]

# do the same for y test filtered
y_test_filtered = y_test.loc[X_test_filtered.index]

# do the same for y
y_filtered = y.loc[X_filtered.index]


# drop rel time from each train set after creating

# create X_train_MW_onset: only keep rows where label isn't self report and relative time is 0
# then drop label & page. Also create groups for CV

# do the same for the filtered x test sets

# datasets of the form X_classifier type are un-split feature sets filtered for 
# each specific classifier type for LOGOCV

X_train_MW_onset = X_train[(X_train["label"] != "self_report") & (X_train["relative_time"] == 0)]
X_train_MW_onset = X_train_MW_onset.copy()
X_train_MW_onset = X_train_MW_onset.drop(columns=["label", "page", "relative_time"])

X_test_filtered_MW_onset = X_test_filtered[(X_test_filtered["label"] != "self_report") & (X_test_filtered["relative_time"] == 0)]
X_test_filtered_MW_onset = X_test_filtered_MW_onset.copy()
X_test_filtered_MW_onset = X_test_filtered_MW_onset.drop(columns=["label", "page", "relative_time"])

# create X_train_MW_onset_2
X_train_MW_onset_2 = X_train[(X_train["label"] != "self_report") & (X_train["relative_time"] == mw2_target)]
X_train_MW_onset_2 = X_train_MW_onset_2.copy()
X_train_MW_onset_2 = X_train_MW_onset_2.drop(columns=["label", "page", "relative_time"])

X_test_filtered_MW_onset_2 = X_test_filtered[(X_test_filtered["label"] != "self_report") & (X_test_filtered["relative_time"] == mw2_target)]
X_test_filtered_MW_onset_2 = X_test_filtered_MW_onset_2.copy()
X_test_filtered_MW_onset_2 = X_test_filtered_MW_onset_2.drop(columns=["label", "page", "relative_time"])

# create X_train_self_report: drop rows where label = mw onset (retain self report and control), then drop label & page
X_train_self_report = X_train[(X_train["label"] != "MW_onset") & (X_train["relative_time"] == (-.5 * window_size))]
X_train_self_report = X_train_self_report.copy()
X_train_self_report = X_train_self_report.drop(columns=["label", "page", "relative_time"]) 

X_test_filtered_self_report = X_test_filtered[(X_test_filtered["label"] != "MW_onset") & (X_test_filtered["relative_time"] == (-.5 * window_size))]
X_test_filtered_self_report = X_test_filtered_self_report.copy()
X_test_filtered_self_report = X_test_filtered_self_report.drop(columns=["label", "page", "relative_time"])

# create classifier specific X subsets for LOGOCV
X_MW_onset = X_filtered[(X_filtered["label"] != "self_report") & (X_filtered["relative_time"] == 0)]
X_MW_onset = X_MW_onset.copy()
X_MW_onset_groups = X_MW_onset["sub_id"]
X_MW_onset = X_MW_onset.drop(columns=["label", "page", "relative_time", "sub_id"])

X_MW_onset_2 = X_filtered[(X_filtered["label"] != "self_report") & (X_filtered["relative_time"] == mw2_target)]
X_MW_onset_2 = X_MW_onset_2.copy()
X_MW_onset_2_groups = X_MW_onset_2["sub_id"]
X_MW_onset_2 = X_MW_onset_2.drop(columns=["label", "page", "relative_time", "sub_id"])

X_self_report = X_filtered[(X_filtered["label"] != "MW_onset") & (X_filtered["relative_time"] == (-.5 * window_size))]
X_self_report = X_self_report.copy()
X_self_report_groups = X_self_report["sub_id"]
X_self_report = X_self_report.drop(columns=["label", "page", "relative_time", "sub_id"]) 

# drop relative time from X_train and X_test now
X_train_relative_times = X_train["relative_time"].copy()
X_train = X_train.drop(columns=["relative_time"])
X_test_relative_times = X_test["relative_time"].copy()
#X_test_relative_times.reset_index(inplace=True, drop=True)
X_test = X_test.drop(columns=["relative_time"])
X_test_filtered_relative_times = X_test_filtered["relative_time"].copy() # now we have relative times for X_test_filtered saved
# each x_test specific to a classifier has been further narrowed down by relative time and label
X_test_filtered = X_test_filtered.drop(columns=["relative_time"])

X_filtered_relative_times = X_filtered["relative_time"].copy()




# drop labels from X_test
X_test = X_test.drop(columns=["label", "page"])

# drop pages and labels from X_train. X_train isn't used again aside from fitting the scaler for the holdout set
# but we need to do this to make that work. relative time has already been dropped
X_train = X_train.drop(columns=["label", "page"])

X_test_filtered = X_test_filtered.drop(columns=["label", "page"])

# don't use X_filtered again, no need to drop

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

# y sets of the form y_classifier type are un-split y sets filtered for each specific
# classifier type for LOGOCV.


# create y_train_MW_onset: 0 for control, 1 for MW_onset when rel time = 0. Rows where label = self_report dropped
y_train_MW_onset = y_train[X_train_relative_times == 0]
y_train_MW_onset = y_train_MW_onset[y_train_MW_onset != "self_report"]
y_train_MW_onset = y_train_MW_onset.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# same for y test filtered
y_test_filtered_MW_onset = y_test_filtered[X_test_filtered_relative_times == 0]
y_test_filtered_MW_onset = y_test_filtered_MW_onset[y_test_filtered_MW_onset != "self_report"]
y_test_filtered_MW_onset = y_test_filtered_MW_onset.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# same for full y
y_MW_onset = y_filtered[X_filtered_relative_times == 0]
y_MW_onset = y_MW_onset[y_MW_onset != "self_report"]
y_MW_onset = y_MW_onset.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_MW_onset_2: : 0 for control, 1 for MW_onset when rel time = 2. Rows where label = self_report dropped
y_train_MW_onset_2 = y_train[X_train_relative_times == mw2_target]
y_train_MW_onset_2 = y_train_MW_onset_2[y_train_MW_onset_2 != "self_report"]
y_train_MW_onset_2 = y_train_MW_onset_2.apply(lambda x: 1 if x in ["MW_onset"] else 0)

y_test_filtered_MW_onset_2 = y_test_filtered[X_test_filtered_relative_times == mw2_target]
y_test_filtered_MW_onset_2 = y_test_filtered_MW_onset_2[y_test_filtered_MW_onset_2 != "self_report"]
y_test_filtered_MW_onset_2 = y_test_filtered_MW_onset_2.apply(lambda x: 1 if x in ["MW_onset"] else 0)

y_MW_onset_2 = y_filtered[X_filtered_relative_times == mw2_target]
y_MW_onset_2 = y_MW_onset_2[y_MW_onset_2 != "self_report"]
y_MW_onset_2 = y_MW_onset_2.apply(lambda x: 1 if x in ["MW_onset"] else 0)

# create y_train_self_report: 0 for control, 1 for self_report. Rows where label = MW_onset dropped
y_train_self_report = y_train[X_train_relative_times == (-.5 * window_size)]
y_train_self_report = y_train_self_report[y_train_self_report != "MW_onset"]
y_train_self_report = y_train_self_report.apply(lambda x: 1 if x in ["self_report"] else 0)

y_test_filtered_self_report = y_test_filtered[X_test_filtered_relative_times == (-.5 * window_size)]
y_test_filtered_self_report = y_test_filtered_self_report[y_test_filtered_self_report != "MW_onset"]
y_test_filtered_self_report = y_test_filtered_self_report.apply(lambda x: 1 if x in ["self_report"] else 0)

y_self_report = y_filtered[X_filtered_relative_times == (-.5 * window_size)]
y_self_report = y_self_report[y_self_report != "MW_onset"]
y_self_report = y_self_report.apply(lambda x: 1 if x in ["self_report"] else 0)

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
X.drop(columns=["label", "page", "relative_time", "sub_id"], inplace=True)


if full_set_relative_times.index.equals(true_labels_full_set.index):
    print("Indexes for full set relative times and true labels full set match!")
else:
    print("Indexes do not match (full set relative times and full set true labels).")

# define models with optimal hps
if window_size == 2:
    self_report_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state), 
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.1), 
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
           # 'Random Forest': RandomForestClassifier(random_state = random_state, n_estimators = 200),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(), 
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(),
            #'XGBoost': XGBClassifier(random_state = random_state)
        }
    
    MW_onset_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state),
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=10),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 20, n_estimators = 100),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(),
            #'XGBoost': XGBClassifier(random_state = random_state)
        }
    
    MW_onset_2_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state),
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.01),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 20, n_estimators = 10),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(),
            #'XGBoost': XGBClassifier(random_state = random_state)
        }
    
if window_size == 5:
    self_report_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=100, solver="lbfgs", penalty = "l2"), 
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=10), 
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 200),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME.R", learning_rate = .1, n_estimators = 100, ),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(), 
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(var_smoothing = .000000001),
            #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=.5, learning_rate = .001, max_depth = 3, n_estimators = 200, subsample=.5)
        }
    
    MW_onset_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=.01, penalty = "l2", solver="liblinear"),
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.01),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 200),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME", learning_rate=.1, n_estimators=50),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(var_smoothing = .000000001),
            #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=1, learning_rate=.001, max_depth=3, n_estimators=100, subsample=.5)
        }
    if mw2_target == 2:
        MW_onset_2_models = {
                'Logistic Regression': LogisticRegression(random_state = random_state, C = .1, penalty = "l2", solver="liblinear"),
                #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.01),
                #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
                #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 100),
                #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME", learning_rate=.1, n_estimators=100),
                #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
                #'KNN': KNeighborsClassifier(),
                #'Naive Bayes': GaussianNB(var_smoothing= .000000001),
                #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=1, learning_rate=.01, max_depth=3, n_estimators=100, subsample=1)
            }
    elif mw2_target == 5:
        MW_onset_2_models = {
                'Logistic Regression': LogisticRegression(random_state = random_state, C = .01, penalty = "l2", solver="liblinear"),
                #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.01),
                #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
                #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 100),
                #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME", learning_rate=.1, n_estimators=100),
                #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
                #'KNN': KNeighborsClassifier(),
                #'Naive Bayes': GaussianNB(var_smoothing= .000000001),
                #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=1, learning_rate=.01, max_depth=3, n_estimators=100, subsample=1)
            }
    elif mw2_target == 2.5:
        MW_onset_2_models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, C=.01, penalty = "l2", solver = "lbfgs")
            }
if window_size == 2:
    self_report_cv_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state),
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.1), 
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
           # 'Random Forest': RandomForestClassifier(random_state = random_state, n_estimators = 200),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(), 
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(),
            #'XGBoost': XGBClassifier(random_state = random_state)
        }
    
    MW_onset_cv_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state),
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=10),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 20, n_estimators = 100),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(),
            #'XGBoost': XGBClassifier(random_state = random_state)
        }
    
    MW_onset_2_cv_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state),
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.01),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 20, n_estimators = 10),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(),
            #'XGBoost': XGBClassifier(random_state = random_state)
        }
    
if window_size == 5:
    self_report_cv_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=100, solver="lbfgs", penalty = "l2"),
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=10), 
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 200),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME.R", learning_rate = .1, n_estimators = 100, ),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(), 
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(var_smoothing = .000000001),
            #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=.5, learning_rate = .001, max_depth = 3, n_estimators = 200, subsample=.5)
        }
    
    MW_onset_cv_models = {
            'Logistic Regression': LogisticRegression(random_state = random_state, C=.01, penalty = "l2", solver="liblinear"),
            #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.01),
            #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
            #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 200),
            #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME", learning_rate=.1, n_estimators=50),
            #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            #'KNN': KNeighborsClassifier(),
            #'Naive Bayes': GaussianNB(var_smoothing = .000000001),
            #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=1, learning_rate=.001, max_depth=3, n_estimators=100, subsample=.5)
        }
    if mw2_target == 2:
        MW_onset_2_cv_models = {
                'Logistic Regression': LogisticRegression(random_state = random_state, C = .1, penalty = "l2", solver="liblinear"),
                #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.01),
                #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
                #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 100),
                #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME", learning_rate=.1, n_estimators=100),
                #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
                #'KNN': KNeighborsClassifier(),
                #'Naive Bayes': GaussianNB(var_smoothing= .000000001),
                #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=1, learning_rate=.01, max_depth=3, n_estimators=100, subsample=1)
            }
    elif mw2_target == 5:
        MW_onset_2_cv_models = {
                'Logistic Regression': LogisticRegression(random_state = random_state, C = .01, penalty = "l2", solver="liblinear"),
                #'Support Vector Machine': SVC(kernel="linear", probability=True, random_state = random_state, C=.01),
                #'Decision Tree': DecisionTreeClassifier(random_state = random_state),
                #'Random Forest': RandomForestClassifier(random_state = random_state, max_depth = 5, n_estimators = 100),
                #'AdaBoost': AdaBoostClassifier(random_state = random_state, algorithm="SAMME", learning_rate=.1, n_estimators=100),
                #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
                #'KNN': KNeighborsClassifier(),
                #'Naive Bayes': GaussianNB(var_smoothing= .000000001),
                #'XGBoost': XGBClassifier(random_state = random_state, colsample_bytree=1, learning_rate=.01, max_depth=3, n_estimators=100, subsample=1)
            }
    elif mw2_target == 2.5:
        MW_onset_2_cv_models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, C=.01, penalty = "l2", solver = "lbfgs")}

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
# adjust pca flag and thresholds here if changes are needed
# reset indices to ensure alignment
X_train_self_report.reset_index(inplace=True, drop=True)
y_train_self_report = y_train_self_report.reset_index(drop=True)

self_report_models, self_report_results, X_test_sr, X_sr, self_report_filtered_results, X_test_filtered_self_report, X_sr_no_PCA = train_models(X_train_self_report, 
                                                                                                      y_train_self_report,
                                                                                                      X_test, X_test_filtered_self_report,
                                                                                                      self_report_models, PCA_flag_self_report,
                                                                                                      random_state,PCA_threshold_self_report, X)
# logocv 
self_report_foldwise_results = run_LOGOCV(X_self_report, y_self_report, X_self_report_groups,
                              self_report_cv_models, "self report", window_size,
                              random_state, PCA_flag_self_report, PCA_threshold_self_report, smote_flag=True)
        
# train MW_onset vs. control 
X_train_MW_onset.reset_index(inplace=True, drop=True)
y_train_MW_onset = y_train_MW_onset.reset_index(drop=True)
MW_onset_models, MW_onset_results, X_test_mw, X_mw, MW_onset_filtered_results, X_test_filtered_MW_onset, X_mw_no_PCA = train_models(X_train_MW_onset,
                                                                                             y_train_MW_onset, X_test,
                                                                                             X_test_filtered_MW_onset,
                                                                                             MW_onset_models, PCA_flag_MW_onset,
                                                                                             random_state, PCA_threshold_MW_onset, X)
    
# logocv
MW_onset_foldwise_results = run_LOGOCV(X_MW_onset, y_MW_onset, X_MW_onset_groups,
                              MW_onset_cv_models, "MW Onset", window_size,
                              random_state, PCA_flag_MW_onset, PCA_threshold_MW_onset, smote_flag=True)
# train MW_onset_2 vs control
X_train_MW_onset_2.reset_index(inplace=True, drop=True)
y_train_MW_onset_2 = y_train_MW_onset_2.reset_index(drop=True)

if mw2_target == 2:
    MW_onset_2_models, MW_onset_2_results, X_test_mw2, X_mw2, MW_onset_2_filtered_results, X_test_filtered_MW_onset_2, X_mw2_no_PCA = train_models(X_train_MW_onset_2,
                                                                                                         y_train_MW_onset_2, X_test,
                                                                                                         X_test_filtered_MW_onset_2,
                                                                                                         MW_onset_2_models, PCA_flag_MW_onset_2,
                                                                                                         random_state, PCA_threshold_MW_onset_2, X)
    # logocv
    MW_onset_2_foldwise_results = run_LOGOCV(X_MW_onset_2, y_MW_onset_2, X_MW_onset_2_groups,
                                  MW_onset_2_cv_models, "MW Onset 2", window_size,
                                  random_state, PCA_flag_MW_onset_2, PCA_threshold_MW_onset_2, smote_flag=True)
elif mw2_target == 5:
    MW_onset_2_models, MW_onset_2_results, X_test_mw2, X_mw2, MW_onset_2_filtered_results, X_test_filtered_MW_onset_2, X_mw2_no_PCA = train_models(X_train_MW_onset_2,
                                                                                                         y_train_MW_onset_2, X_test,
                                                                                                         X_test_filtered_MW_onset_2,
                                                                                                         MW_onset_2_models, PCA_flag_MW_onset_5,
                                                                                                         random_state, PCA_threshold_MW_onset_5, X)
    MW_onset_2_foldwise_results = run_LOGOCV(X_MW_onset_2, y_MW_onset_2, X_MW_onset_2_groups,
                                  MW_onset_2_cv_models, "MW Onset 2", window_size,
                                  random_state, PCA_flag_MW_onset_5, PCA_threshold_MW_onset_5, smote_flag=True)
elif mw2_target == 2.5:
    MW_onset_2_models, MW_onset_2_results, X_test_mw2, X_mw2, MW_onset_2_filtered_results, X_test_filtered_MW_onset_2, X_mw2_no_PCA = train_models(X_train_MW_onset_2,
                                                                                                         y_train_MW_onset_2, X_test,
                                                                                                         X_test_filtered_MW_onset_2,
                                                                                                         MW_onset_2_models, PCA_flag_MW_onset_2_5,
                                                                                                         random_state, PCA_threshold_MW_onset_2_5, X)
    # logocv
    MW_onset_2_foldwise_results = run_LOGOCV(X_MW_onset_2, y_MW_onset_2, X_MW_onset_2_groups,
                                  MW_onset_2_cv_models, "MW Onset 2", window_size,
                                  random_state, PCA_flag_MW_onset_2_5, PCA_threshold_MW_onset_2_5, smote_flag=True)


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
plot_auroc_f1(filtered_aurocs, filtered_f1s, window_size, mw2_target, no_mw2_flag, filtered=True)

# display auroc and f1 in grouped bar chart 
plot_auroc_f1(aurocs, f1s, window_size, mw2_target, no_mw2_flag, filtered=False)

# plot probabilities from self_report on holdout set and mw_onset on holdout set
# feed this the true_labels_test_set so it has different indicators for self_report and mw_onset

#plot_probs(test_results, X_test_relative_times, true_labels_test_set, window_size, CV)
"""
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
"""

raw_scores = get_raw_scores(MW_onset_models, self_report_models, MW_onset_2_models,
                            X_mw, X_mw2, X_sr, mw2_target, full_set_relative_times,
                            true_labels_full_set, window_size, feature_importance_flag = False)
plot_raw_scores(full_set_relative_times, true_labels_full_set, window_size,
                raw_scores, no_mw2_flag, mw2_target, trial_count_df, mw_color, sr_color, mw2_color)




feat_importance_raw_scores = get_raw_scores(MW_onset_models, self_report_models, MW_onset_2_models, 
                                            X_mw, X_mw2, X_sr, mw2_target, full_set_relative_times,
                                            true_labels_full_set, window_size, feature_importance_flag = True)
plot_forward_feature_importance_reformatted(feat_importance_raw_scores, window_size, X_mw_no_PCA,
                                X_mw2_no_PCA, X_sr_no_PCA, full_set_relative_times,
                                true_labels_full_set, no_mw2_flag, mw2_target,mw_color, sr_color, mw2_color)

# plot predictor hists 

#predictor_hist(raw_scores, true_labels_full_set, "mw_onset", window_size, mw2_target, no_mw2_flag)
#predictor_hist(raw_scores, true_labels_full_set, "self_report", window_size, mw2_target, no_mw2_flag)
#if no_mw2_flag == False: # only plot mw2 predictor hist if that's on in this pipeline
    #predictor_hist(raw_scores, true_labels_full_set, "mw_onset_2", window_size, mw2_target, no_mw2_flag)
    
    
# display roc curves for foldwise results
# we have y scores for each fold stored in classifier_type_foldwise_results.
# each list idx corresponds to a fold. Model name is key (all logistic regression)
# then y scores and true labels are secondary keys

# plot curve for each classifier type - these are teh cross validation results
plot_roc(self_report_foldwise_results, "Self Report", window_size, model_names, mw_color, sr_color, mw2_color)
plot_roc(MW_onset_foldwise_results, "MW Onset", window_size, model_names, mw_color, sr_color, mw2_color)
plot_roc(MW_onset_2_foldwise_results, f"MW Onset {mw2_target}", window_size, model_names, mw_color, sr_color, mw2_color)

# calculate and plot foldwise auroc
if no_mw2_flag == False:
    plot_foldwise_auroc_histogram(MW_onset_foldwise_results, self_report_foldwise_results,
                        window_size, model_names, mw_color, sr_color, mw2_color,
                        mw2_target, MW_onset_2_foldwise_results)
else:
    plot_foldwise_auroc_histogram(MW_onset_foldwise_results, self_report_foldwise_results,
                        window_size, model_names, mw_color, sr_color, mw2_color)


