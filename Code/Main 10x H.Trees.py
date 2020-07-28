# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:12:16 2020

@author: JZM
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:08:18 2020

@author: JZM
"""

# Imports
import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
import pandas as pd
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
import glob
import os
from skmultiflow.meta import BatchIncrementalClassifier

all_Files = r'C:\Users\JZM\Desktop\concept-drift-master\concept-drift-master\data\\'
all_Datasets = glob.glob(all_Files +'*.csv')
Results = []
for x in all_Datasets:
    print("Results for "+ str(os.path.basename(x)))
    try:
    #    Importing Data as stream
        data_stream = FileStream(x)    
        data_stream.prepare_for_use()
#        awe = AccuracyWeightedEnsembleClassifier(n_estimators=10,
#                                                base_estimator=HoeffdingTreeClassifier)
       
    #    Defining Classifier     HT is simple where Adaptive H.Tree uses adwin to minimze error
        CLF = [HoeffdingAdaptiveTreeClassifier()]
    
    #    Defininf and Training      
        eval = EvaluatePrequential(show_plot=True,
                                   metrics=['accuracy','kappa','model_size','precision','recall','f1'],
                                   n_wait=100)    
        eval.evaluate(stream=data_stream, model = CLF, model_names  = ['HAT'])
        
        clf = BatchIncrementalClassifier(base_estimator=HoeffdingTreeClassifier(), n_estimators=10)
        # Keeping track of sample count and correct prediction count
        sample_count = 0
        corrects = 0
        # Pre training the classifier with 200 samples
        X, y = data_stream.next_sample(200)
        clf = clf.partial_fit(X, y, classes=data_stream.target_values)
        for i in range(2000):
            X, y = data_stream.next_sample()
            pred = clf.predict(X)
            clf = clf.partial_fit(X, y)
            if pred is not None:
                if y[0] == pred[0]:
                    corrects += 1
            sample_count += 1
            
            
        # Displaying the results
        print(str(sample_count) + ' samples analyzed.')
        print('OzaBaggingClassifier Accuracy: ' + str(corrects / sample_count))

        print("********************************")
        print("Results for "+ str(os.path.basename(x)))
        print("********************************")
    except:
        print("Error......!!")


# Imports
#
#
#
#clf = BatchIncrementalClassifier(base_estimator=HoeffdingTreeClassifier(), n_estimators=10)
## Keeping track of sample count and correct prediction count
#sample_count = 0
#corrects = 0
## Pre training the classifier with 200 samples
#X, y = data_stream.next_sample(200)
#clf = clf.partial_fit(X, y, classes=data_stream.target_values)
#for i in range(2000):
#    X, y = data_stream.next_sample()
#    pred = clf.predict(X)
#    clf = clf.partial_fit(X, y)
#    if pred is not None:
#        if y[0] == pred[0]:
#            corrects += 1
#    sample_count += 1
#    
#    
## Displaying the results
#print(str(sample_count) + ' samples analyzed.')
#print('OzaBaggingClassifier Accuracy: ' + str(corrects / sample_count))













