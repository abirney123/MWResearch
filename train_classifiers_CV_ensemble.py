#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:43:01 2025

@author: Alaina

Train with CV then use averaged y_scores from all folds to generate final
classifications.

Plot ROC curves, predictor histograms for ensemble models
To improve:
    - weighted averaging of y_scores depending on fold auroc
    - investigate if scaling approach is optimal
    - fix docstrings
    - try with and without PCA
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
from sklearn.decomposition import PCA, KernelPCA

def load_extract_preprocess(filepath, features, random_state=42):
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

def evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """
    Trains and evaluates a model, calculating ROC-AUC, F1 score, confusion matrix,
    predicted probabilities, tpr, and fpr. Also returns the true labels for the 
    test set. If the model being evaluated is a random forest, the feature 
    importances will also be returned.

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
    y_pred : Array of float.
        Predicted probabilities.
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
    
    if model_name == "Random Forest":
        importances = model.feature_importances_
        return y_scores, roc_auc, importances # only return importances if RF
    else:
        return y_scores, roc_auc

def predictor_hist(results, y_test):
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
    fig.suptitle("Predictor Histograms", fontsize=16)
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
    plt.savefig(f"Predictor_hist_ensemble.png")
    plt.close()
    
    # KDE
    # one figure with a subplot for each model
    # 9 models, arrange in 3x3 grid
    fig, axes = plt.subplots(3,3,figsize=(15,15))
    fig.suptitle("Predictor Histograms", fontsize=16)
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
    plt.savefig(f"Predictor_hist_ensemble_KDE.png")
        
def plot_roc(results, y_test):
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

    plt.title('ROC Curves')
    plt.savefig("ROC_curves_ensemble.png")
    plt.show()
    
def avg_feat_importance(results, columns):
    """
    Finds and plots random forest feature importance, averaged over folds. The 
    resulting plot is saved. Code adapted from HS.

    Parameters
    ----------
    results : Dictionary
        Nexted dictionary of results from LOPOCV model training and evaluation. 
        First level keys are model names. Second level keys are as follows:
        roc_auc, f1, conf_matrix, true_labels, predicted_probs, tpr, fpr, importances.
    columns : Index
        Columns/ features as ordered in the training data.


    Returns
    -------
    None.

    """
    # Get feature importances for all folds
    importances = results["Random Forest"]["importances"]
    # average importances over folds
    importances = np.mean(importances, axis=0)

    indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    plt.title("Random Forest Feature Importance- Averaged Over Folds")
    plt.bar(range(len(columns)), importances[indices], align="center")
    plt.xticks(range(len(columns)), [columns[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.savefig("feat_importance_avg_experiment.png")
    plt.show()
    
def plot_conf_mat(results, y_test, model_names):
    """
    Plots confusion matrix heatmaps for up to four models. One figure with a subplot
    for each specified model will be created. The figure is also saved
    to the current working directory.

    Parameters
    ----------
    results : Dictionary
        Nested dictionary of results from ensemble model evaluation. This 
        dictionary should hold y_scores averaged over CV models and the 
        corresponding predictions.
    y_test: Array of bool.
        True labels
    model_names: List of str.
        The models to plot confusion matrices for.


    Returns
    -------
    None.

    """
    # use ensemble results - results from y_scores averaged over cv models
    # one figure with a subplot for each model name in list
    num_models = len(model_names)
    fig, axes = plt.subplots(2,2, figsize=(12,12))
    axes = axes.flatten()
    for subplot_idx, model_name in enumerate(model_names):
        predictions = results[model_name]["predictions"]
        conf_mat = confusion_matrix(y_test, predictions)
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

    plt.savefig("ensemble_conf_matrices.png")
    plt.show()
    

# load data
random_state = 42
weighted = True
PCA_flag = False
pca_threshold = .97

filepath = "group_R_features_same-dur.csv"
# Features (HS): pupil slope, norm pupil, norm_fix_word_num,
# norm_sac_num, norm_in_word_reg, norm_out_word_reg, zscored_zipf_fixdur_corr,
# norm_total_viewing, zscored_word_length_fixdur_corr
# me: including fix_num, blink_num, blink_dur, blink_freq because they don't seem to 
# be represented here outright or through normalized values. Removing fix_num
# and blink_freq bc highly correlated with blink_num
"""
features = ["pupil_slope", "norm_pupil", "norm_fix_word_num", "norm_sac_num", 
            "norm_in_word_reg", "norm_out_word_reg", "zscored_zipf_fixdur_corr",
            "norm_total_viewing", "zscored_word_length_fixdur_corr", "blink_num",
            "blink_dur"]
"""
features = ["pupil_slope", "norm_pupil", "norm_fix_word_num", "norm_sac_num", 
            "norm_in_word_reg", "norm_out_word_reg", "zscored_zipf_fixdur_corr",
            "norm_total_viewing", "zscored_word_length_fixdur_corr", 
            "blink_dur"]
# get train and test features and labels - X_test, y_test are holdout set
X_train, X_test, y_train, y_test, groups, columns = load_extract_preprocess(filepath, features, random_state)
#print(type(X_train))


logo = LeaveOneGroupOut()

# define models- logistic regression, SVM, decision tree, random forest, Adaboost,
# LDA, KNN, Naive Bayes, XGBoost
models = {
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

# train with CV, get predicted probs and auroc and then evaluate on holdout set
cv_results = {model_name:{
    "predicted_probs" : [],
    "roc_auc" : [],
    "importances" : [] if model_name == "Random Forest" else None# store auroc in case we want to do a weighted average
    }for model_name in models.keys()}

# results of individual cv models evaluation on holdout set
test_results = {model_name:{
    "predicted_probs" : []
    }for model_name in models.keys()}


# scale holdout set for later
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit on full train set, transform but store for only pca, not cv (need to scale for each fold)
X_test = scaler.transform(X_test)

fold_idx = 1
# LOGOCV loop

if PCA_flag == True:
    # dimensionality reduction/ feature selection
    # find top k components for the full set (explain ~x% of variance)
    # train PCA on train set, transform test set
    pca = PCA(random_state = random_state)
    pca.fit(X_train_scaled) # just fitting, not transforming at this point because want to investigate how many components to keep
    variance_ratio = pca.explained_variance_ratio_
    #print(variance_ratio)
    threshold = pca_threshold
    cumsum = 0
    rat_idx = 0 # index for variance ratio list
    while cumsum < threshold:
        cumsum += variance_ratio[rat_idx]
        rat_idx += 1
        
    print(f"{rat_idx} components explain {cumsum * 100: .2f}% of variance")
    
    
    # reduce data to num components that explain ~ threshold variance
    num_components = rat_idx 
    reduce_pca = PCA(n_components = num_components, random_state = random_state)
    # fit to full train set
    reduce_pca.fit(X_train_scaled)
    # transform holdout set
    X_test = reduce_pca.transform(X_test)

for train_idx, test_idx in logo.split(X_train,y_train,groups):
    #print(type(X_train))
    # train test split for CV
    X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]
    
    #print(f"Fold {fold_idx}, length {len(X_train)}")

    
    #print("Class counts in train dataset")
    #MR_count = y_train.sum()
    #NR_count = len(y_train) - MR_count
    #print(f"MR: {MR_count}, NR: {NR_count}")
    
    # scale features for this fold
    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_test_fold = scaler.transform(X_test_fold)
    
    if PCA_flag == True:
        # PCA transformation on features for this fold
        X_train_fold = reduce_pca.transform(X_train_fold)
        X_test_fold = reduce_pca.transform(X_test_fold)

    for model_name, model in models.items():
        # evaluate is called for each fold, so metrics generated for each fold
            if model_name == "Random Forest":
                predicted_probs, auroc, importances = evaluate(model,X_train_fold,y_train_fold,X_test_fold,
                                          y_test_fold, model_name)
            else:
                predicted_probs, auroc = evaluate(model,X_train_fold,y_train_fold,X_test_fold,
                                          y_test_fold, model_name)
        
            cv_results[model_name]["predicted_probs"].append(predicted_probs)
            cv_results[model_name]["roc_auc"].append(auroc)
            if model_name == "Random Forest":
                cv_results[model_name]["importances"].append(importances)
            # get y_scores on holdout set for ensemble predictions
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:,1]
            else:
                y_scores = model.decision_function(X_test)
            test_results[model_name]["predicted_probs"].append(y_scores)

# ensemble results - holds y_probs averaged over cv model folds
ensemble_results = {model_name:{
    "avg_probs" : None
    }for model_name in models.keys()}


if weighted == True: # use weighted arithmetic mean
    # loop through model types
    for model_name, model in models.items():
        # get auc for each fold's model's test set evaluation
        fold_aurocs = [] # 1 auroc for each fold
        all_folds_y_scores = test_results[model_name]["predicted_probs"]
        for fold_scores in all_folds_y_scores:
            fold_auc = roc_auc_score(y_test, fold_scores)
            fold_aurocs.append(fold_auc)
            
        # get info for weighted arithmetic mean of aurocs and intercepts/ weights
        # normalization not needed due to weighted arithmetic mean
        averaging_weights = fold_aurocs #np.array(fold_aurocs) / np.sum(fold_aurocs)
        weighting_denominator = np.sum(averaging_weights)
        
        # weighted arithmetic mean of y_scores over folds dependent on auc
        all_fold_probs_arr = np.array(test_results[model_name]["predicted_probs"]) # rows are folds, cols are samples
        weighted_scores_numerator = np.zeros_like(all_fold_probs_arr) 
    
        # loop through each fold for this model type
        for fold_idx, fold_scores in enumerate(test_results[model_name]["predicted_probs"]):
            # mult. the auc for this fold by the scores for this fold's model on test set
            weighted_scores_numerator += averaging_weights[fold_idx] * fold_scores
            
        weighted_scores = weighted_scores_numerator / weighting_denominator
        ensemble_results[model_name]["avg_probs"] = weighted_scores[0,:] # grab the first row bc they get repeated
else:
    # average the test y_scores over folds
    # for each model, test results y_scores is a list with one entry for each fold. Each entry holds an array of scores
    # so get the avearges over the same indices in each of the lists for each model
    for model_name, model in models.items():
        all_fold_probs_list = test_results[model_name]["predicted_probs"]
        all_fold_probs_arr = np.array(all_fold_probs_list) # num folds x samples - get average over rows
        avg_probs = np.mean(all_fold_probs_arr, axis=0)
        # now there is one probability for each sample for current model
        ensemble_results[model_name]["avg_probs"] = avg_probs
    

predict_threshold = .5
# turn into classifications for each model
for model_name in models.keys():
    avg_probs = ensemble_results[model_name]["avg_probs"]
    ensemble_results[model_name]["predictions"] = (avg_probs >= predict_threshold).astype(int)



# plot roc curves with auroc
plot_roc(ensemble_results, y_test)

# predictor histograms
predictor_hist(ensemble_results, y_test)

# feature importance
avg_feat_importance(cv_results, columns)

# confusion matrices for LDA, SVM, logistic regression (highest performers)
model_names = ["Logistic Regression", "Support Vector Machine", "Linear Discriminant Analysis"]
plot_conf_mat(ensemble_results, y_test, model_names)
