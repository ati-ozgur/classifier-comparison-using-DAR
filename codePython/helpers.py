from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer,roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from time import time
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import OrdinalEncoder
import joblib
import numpy as np

import six
import sys
sys.modules['sklearn.externals.six'] = six
sys.modules['sklearn.externals.joblib'] = joblib


from pathlib import Path



def get_kdd99_10percent():
    return _get_kdd99(full=False)

def get_kdd99_full():
    return _get_kdd99(full=True)

def _get_kdd99(full=False):
    percent10 = not full
    (X,y) = sklearn.datasets.fetch_kddcup99(return_X_y=True,percent10=percent10)
    enc = OrdinalEncoder()
    enc.fit(X[:,[1,2,3]])
    X[:,[1,2,3]] = enc.transform(X[:,[1,2,3]])
    X = X.astype(np.float)

    enc_y = OrdinalEncoder()
    enc_y.fit(y.reshape(-1, 1))

    y_new = enc_y.transform(y.reshape(-1, 1))

    return (X,y_new)

# https://scikit-learn.org/stable/supervised_learning.html
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Add according to need
list_of_classifiers = [
    {"Name": "One Rule", "Classifier": DummyClassifier(strategy="stratified")},
    {"Name": "Zero Rule", "Classifier": DummyClassifier(strategy="most_frequent")},
    {"Name": "Naive Bayes ", "Classifier" :GaussianNB()},
    {"Name": "AdaBoost", "Classifier" :AdaBoostClassifier()},
    {"Name": "Bagging", "Classifier" :BaggingClassifier()},
    {"Name": "Decision Tree (CART)", "Classifier" :DecisionTreeClassifier()},
    {"Name": "K Neighbors", "Classifier" :KNeighborsClassifier()},
    {"Name": "Logistic Regression", "Classifier" :LogisticRegression()},
    {"Name": "Multi Layer Perceptron  ", "Classifier" :MLPClassifier()},
    {"Name": "Random Forest", "Classifier" :RandomForestClassifier()},
    {"Name": "Support Vector Machines  ", "Classifier" :SVC()},
    ]

def train_model(X_train, X_test, Y_train, Y_test, classifier_name,classifier,cv_current,cv_count) :
    row_size = X_train.shape[0] + Y_train.shape[0]
    ret_metrics = {}
    t_training_start = time()
    classifier.fit(X_train, np.ravel(Y_train))
    t_training_end = time() 
    ret_metrics["training_time"] = t_training_end- t_training_start
    ret_metrics["t_training_start"] = t_training_start
    ret_metrics["t_training_end"] = t_training_end
    t_testing_start = time()
    Y_test_pred = classifier.predict(X_test)
    t_testing_end = time()
    ret_metrics["testing_time"] = t_testing_end - t_testing_start
    ret_metrics["t_testing_start"] = t_testing_start
    ret_metrics["t_testing_end"] = t_testing_end

    ret_metrics["accuracy_score"] = accuracy_score(Y_test,Y_test_pred) 
    ret_metrics["precision_score"] = precision_score(Y_test,Y_test_pred,average='micro') 
    ret_metrics["recall_score"] = recall_score(Y_test,Y_test_pred,average='micro') 
    ret_metrics["f1_score"] = f1_score(Y_test,Y_test_pred,average='micro') 
    ret_metrics["confusion_matrix"] = confusion_matrix(Y_test,Y_test_pred) 
    #ret_metrics["roc_auc_score"] = roc_auc_score(Y_test,Y_test_pred,average='micro',multi_class="ovo") 
    ret_metrics["classifier_name"] = classifier_name
    ret_metrics["cross_validation_count"] = cv_current



    model_filename = f"model_files/{classifier_name}_size_{row_size}_cv_{cv_current}_{cv_count}.sav"
    joblib.dump(classifier, model_filename)
    ret_metrics["model_size"] = Path(model_filename).stat().st_size



    return ret_metrics


