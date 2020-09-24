import numpy as np
import pandas as pd
import os
import sys


import sklearn
import sklearn.datasets

from helpers import *


def run_experiment_once(classifier_name,classifier,k_fold,X,y):
    accuracy_score_sum = 0.0
    training_time_sum = 0.0 
    testing_time_sum = 0
    model_size_sum = 0
    for k, (train, test) in enumerate(k_fold.split(X, y)):
        print(f"classifier name {classifier_name},cross validation count {k}")
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        ret_metrics = train_model(X_train, X_test, y_train, y_test,classifier_name, classifier,k,CV_COUNT)
        print(ret_metrics)
        accuracy_score_sum += ret_metrics["accuracy_score"]
        training_time_sum += ret_metrics["training_time"]
        testing_time_sum += ret_metrics["testing_time"]
        model_size_sum += ret_metrics["model_size"]

    mean_accuracy_score = accuracy_score_sum / CV_COUNT
    mean_training_time = training_time_sum / CV_COUNT
    mean_testing_time = testing_time_sum / CV_COUNT
    mean_model_size = model_size_sum / CV_COUNT


    result_dictionary = {}
    result_dictionary["classifier_name"] = classifier_name    
    result_dictionary["mean_accuracy_score"] = mean_accuracy_score    
    result_dictionary["mean_training_time"] = mean_training_time    
    result_dictionary["mean_testing_time"] = mean_testing_time    
    result_dictionary["mean_model_size"] = mean_model_size    


    return result_dictionary


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
            result_dictionary = run_experiment_once(classifier_name,clf,k_fold,X,y)
            df = get_dataframe(df_filename)
            df = df.append(result_dictionary,ignore_index=True)
            print(result_dictionary)
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
