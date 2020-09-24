import numpy as np
import pandas as pd
import os
import sys


import sklearn
import sklearn.datasets

from helpers import *
from pprint import pprint



print("Common name |  Scikit-learn class name |  Default parameters")
for dic in list_of_classifiers:
    output_line = ""

    name = dic["Name"]
    classifier = dic["Classifier"]
    #print(classifier)
    output_line = output_line+ f"{name} |  {type(classifier).__name__} | "
    clf_variables = vars(classifier)
    #print(type(clf_variables))
    #print(clf_variables)
    for key in sorted(clf_variables):
        output_line = output_line+ f"{key}:{clf_variables[key]},"

    print(output_line)