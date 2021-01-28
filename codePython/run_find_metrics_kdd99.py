import numpy as np
import pandas as pd
import os
import sys


import sklearn
import sklearn.datasets

from helpers import *


def run_experiment_once(classifier_name,classifier,k_fold,X,y):
    l_results = []
    for k, (train, test) in enumerate(k_fold.split(X, y)):

        print(f"classifier name {classifier_name},cross validation count {k}")
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        ret_metrics = train_model(X_train, X_test, y_train, y_test,classifier_name, classifier,k,CV_COUNT)
        print(ret_metrics)

        l_results.append(ret_metrics)



    return l_results


def get_dataframe(df_filename):
    if os.path.isfile(df_filename):
        df = pd.read_csv(df_filename)
    else:
        df_columns = ["classifier_name","mean_accuracy_score","mean_training_time","mean_testing_time","mean_model_size"]
        df = pd.DataFrame(columns=df_columns)

    return df




def run_experiment(dataset_type,CV_COUNT = 3,clf_index = None):

    if clf_index is None:
        list_clf = list_of_classifiers
    else:
        list_clf = list_of_classifiers[clf_index:clf_index+1]

    df_filename = f"results/kdd99_{dataset_type}_{CV_COUNT}_fold.csv"

    if (dataset_type == "full"):
        X,y = get_kdd99_full()
    elif (dataset_type == "10percent"):
        X,y = get_kdd99_10percent()


    k_fold = KFold(n_splits=CV_COUNT, random_state=True, shuffle=True)
    k_fold.get_n_splits(X)


    for classifier_dict in list_clf:

        df = get_dataframe(df_filename)
        #print(df.shape)
        #print(df.classifier_name.values)
        classifier_name = classifier_dict['Name']

        if classifier_name in df.classifier_name.values:
            print(f"results exists for {classifier_name}")
        else:
            clf = classifier_dict["Classifier"]
            result = run_experiment_once(classifier_name,clf,k_fold,X,y)
            df = get_dataframe(df_filename)
            df = df.append(result,ignore_index=True)
            print(result)
            df.to_csv(df_filename,index=False)



if __name__=="__main__":

    if len(sys.argv) > 1:
        clf_index = int(sys.argv[1])
    else:
        clf_index = None

    CV_COUNT = 10
    #dataset_type = "full"
    dataset_type = "10percent"

    run_experiment(dataset_type,CV_COUNT,clf_index)
