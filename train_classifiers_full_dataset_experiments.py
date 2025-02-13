#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:16:46 2025

@author: Alaina
* if want to optimize, don't save true labels in results. Y_test can just be used 
since no CV. That logic is left over from other script
* random state for comparability/ reproducablitiy
"""
# Imports and functions

import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, f1_score, silhouette_score
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
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import umap

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = random_state)
    
    columns = X.columns
    """
    # check for multicollinearity
    corr_matrix = pd.DataFrame(X, columns=columns).corr()
    plt.figure(figsize=(14,15))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.xticks(rotation=45)
    plt.title("Correlation Matrix of Features")
    plt.tight_layout()
    plt.savefig("features_corr_matrix_experimental.png")
    plt.show()
    """

    return X_train, X_test, y_train, y_test, columns

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
    roc_auc : Float.
        Area under the ROC curve.
    f1 : Float.
        The F1 score.
    conf_matrix : Array of int.
        The confusion matrix.
    y_test : Array of bool.
        True labels for the test set.
    y_pred : Array of float.
        Predicted probabilities.
    tpr : Array of float.
        True positive rate/ sensitivity/ recall. The proportion of instances
        of the positive class correctly identified as postive instances.
    fpr : Array of float.
        False positive rate. The proportion of instances of the negative class
        that are incorrectly identified as positive instances.
    importances: Array of float.
        Feature importances. Only returned if the model being evaluated is 
        a random forest.
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
    # get f1
    f1 = f1_score(y_test, y_pred)
    # get conf matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    #print(f"y_test dtype in evaluate: {y_test.dtype}")
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    # if random forest model, return feature importance
    if model_name == "Random Forest":
        importances = model.feature_importances_
        return roc_auc, f1, conf_matrix, y_test, y_scores, tpr, fpr, importances
    else: 
        return roc_auc, f1, conf_matrix, y_test, y_scores, tpr, fpr
    
def predictor_hist(results):
    """
    
    Plot and saves a histogram of the predictor variable for each model. One figure
    is created with a subplot for each model type.

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
    fig.suptitle(f"Predictor Histograms", fontsize=16)
    # flatten axes
    axes = np.array(axes).flatten()
    # get y_pred for each model
    for idx, (model_name, metrics) in enumerate(results.items()):

        predictions = metrics["predicted_probs"]
        labels = metrics["true_labels"]

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
    plt.savefig(f"Predictor_hist_noCV.png")
        
def plot_roc(results):
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
        fpr = metrics["fpr"][0]
        tpr = metrics["tpr"][0]
        auc = metrics["roc_auc"][0]

        
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
    plt.savefig("ROC_curves_noCV.png")
    plt.show()
    
def feat_importance(results, columns):
    """
    Plots random forest feature importance.The 
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
    # Get feature importances for all folds and extract from list
    importances = results["Random Forest"]["importances"]
    importances = importances[0]

    indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order

    # Plot feature importance
    plt.figure(figsize=(12, 10))
    plt.title("Random Forest Feature Importance")
    plt.bar(range(len(columns)), importances[indices], align="center")
    plt.xticks(range(len(columns)), [columns[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.savefig("feat_importance_noCV.png")
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
        conf_mat = results[model_name]["conf_matrix"][0]
        
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
    
def forward_model(model_names, models, X_test):
    """
    Calculate forward model from weights of specified models and plot feature
    importance from those forward models in a grid of subplots. Can create plots
    for up to four models at once.
    
    Parameters:
    ----------
    model_name: List of Str, required.
        The names of the models to compute the forward models for and plot corresponding
        feature importance. Only linear models should be specified here as they
        have the coef attribute.
    models: Dict, required.
        A dictionary holding pre-trained models. Must contain the model name specified
        through model_name as a key.
    X_test: DataFrame, required.
        Dataframe of test features. Must have column names.
        
    Returns:
    -------
    None.
        
    """
    
    num_models = len(model_names)
    
    fig, axes = plt.subplots(2,2, figsize=(15,15))
    axes = axes.flatten()
    
    for subplot_idx, model_name in enumerate(model_names):

        # find Zj = Wj^T * Xj : https://iopscience.iop.org/article/10.1088/1741-2560/11/4/046003 2.5 (2) but without binning
        # then forward model Aj = (XjZj) / (Zj^T Zj) 
        
        # get the weights
        if hasattr(models[model_name], "coef_"):
            w = models[model_name].coef_.flatten() # list, one entry for each feature
        else:
            print("The specified model doesn't have the coef_ attribute...")
            
        # find Z - dot prod of X_test and weights
        Z = np.dot(X_test, w)  # array, one entry for each sample
        
        # find A
        A_numerator = np.dot(X_test.T, Z) # transpose X so inner dimensions match for mult.
        A_denom = np.dot(Z.T, Z)
        A = A_numerator / A_denom # now A has one importance value for each feature
        
        
        # importance A[0] corresponds to feature 0 in X_test
        feature_names = X_test.columns
            
        # plot
        # Get feature importances for all folds and extract from list
        importances = np.abs(A)
    
        indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order
    
        # Plot feature importance
        ax = axes[subplot_idx]
        ax.set_title(f"{model_name}")
        ax.bar(range(len(feature_names)), importances[indices], align="center")
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_ylabel('Weight')
        
    # remove empty subplots 
    for j in range(subplot_idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Forward Model Feature Importance (Absolute Values)")
    plt.savefig(f"forward_feat_importance.png")
    plt.show()
    
#%%
# load in & split data
filepath = "group_R_features_same-dur.csv"
random_state = 42
PCA_flag = False
PCA_threshold = .8


# full set
"""
features = ["pupil_slope", "norm_pupil", "norm_fix_word_num", "norm_sac_num", 
            "norm_in_word_reg", "norm_out_word_reg", "zscored_zipf_fixdur_corr",
            "norm_total_viewing", "zscored_word_length_fixdur_corr", "blink_num",
            "blink_dur"]
"""

# BFE_RR
"""
features = ["pupil_slope", "norm_pupil", "norm_fix_word_num", "norm_sac_num", 
            "norm_out_word_reg", "zscored_zipf_fixdur_corr","blink_dur"]
"""


X_train, X_test, y_train, y_test, columns = load_extract_preprocess(filepath, features)



# scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if PCA_flag:
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
    
    
    # reduce data to num components that explain ~ threshold variance
    num_components = rat_idx 
    reduce_pca = PCA(n_components = num_components, random_state = random_state)
    # fit transform train
    X_train = reduce_pca.fit_transform(X_train)
    # transform test
    X_test = reduce_pca.transform(X_test)
    
    
    # X_train, test cols now represent components rather than features
    columns = []
    for i in range(num_components):
        columns.append(f"component_{i}")

# back to df after scaling
X_train = pd.DataFrame(X_train, columns = columns)
X_test = pd.DataFrame(X_test, columns = columns)

# train models

# define models- logistic regression, SVM, decision tree, random forest, Adaboost,
# LDA, KNN, Naive Bayes, XGBoost
models = {
        'Logistic Regression': LogisticRegression(class_weight="balanced", random_state = random_state, penalty="l1", solver="liblinear"),
        'Support Vector Machine': SVC(kernel="linear", probability=True, class_weight="balanced", random_state = random_state),
        'Decision Tree': DecisionTreeClassifier(class_weight="balanced", random_state = random_state),
        'Random Forest': RandomForestClassifier(class_weight="balanced", random_state = random_state),
        'AdaBoost': AdaBoostClassifier(random_state = random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier()
    }


# set up data storage structure - nested dicts, key for each model type with a dictionary
# of results for that model within it
# importances will only be populated for random forest
results = {model_name:{
    "roc_auc" : [],
    "f1" : [],
    "conf_matrix" : [],
    "true_labels" : [],
    "predicted_probs" : [],
    "tpr" : [],
    "fpr" : [],
    "importances" : [] if model_name == "Random Forest" else None
    }for model_name in models.keys()}

# train and test models

for model_name, model in models.items():
    # evaluate is called for each fold, so metrics generated for each fold
    if model_name == "Random Forest":
        roc_auc, f1, conf_matrix, true_labels, predicted_probs, tpr, fpr, importances = evaluate(model,
                                                                          X_train,
                                                                          y_train,
                                                                          X_test,
                                                                          y_test,
                                                                          model_name)
    else:
        roc_auc, f1, conf_matrix, true_labels, predicted_probs, tpr, fpr = evaluate(model,
                                                                          X_train,
                                                                          y_train,
                                                                          X_test,
                                                                          y_test,
                                                                          model_name)
    

    results[model_name]["roc_auc"].append(roc_auc)
    results[model_name]["f1"].append(f1)
    results[model_name]["conf_matrix"].append(conf_matrix)
    results[model_name]["true_labels"].append(true_labels)
    results[model_name]["predicted_probs"].append(predicted_probs)
    results[model_name]["tpr"].append(tpr)
    results[model_name]["fpr"].append(fpr)
    if model_name == "Random Forest":
        results[model_name]["importances"].append(importances)


# predictor histograms
predictor_hist(results)

# roc plot
plot_roc(results)
# rf feature importance
feat_importance(results, columns)


model_names = ["Logistic Regression", "Support Vector Machine", "Linear Discriminant Analysis"]
plot_conf_mat(results, y_test, model_names)

# forward model feature importance for linear models, these are the same as our top three performers which is nice
linear_models = ["Logistic Regression", "Support Vector Machine", "Linear Discriminant Analysis"]
# defined forward model for each linear model & plot feature importance
forward_model(linear_models, models, X_test)

