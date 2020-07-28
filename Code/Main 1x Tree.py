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



all_Files = r'data\\'
all_Datasets = glob.glob(all_Files +'*.csv')
Results = []
for x in all_Datasets:
    print("Results for "+ str(os.path.basename(x)))
    try:
    #    Importing Data as stream
        data_stream = FileStream(x)    
        data_stream.prepare_for_use()
        
    #    Defining Classifier     HT is simple where Adaptive H.Tree uses adwin to minimze error
        CLF = [HoeffdingTreeClassifier(),HoeffdingAdaptiveTreeClassifier()]
    
    #    Defininf and Training      
        eval = EvaluatePrequential(show_plot=True,
                                   metrics=['accuracy','kappa','model_size','precision','recall','f1'],
                                   n_wait=100)    
        eval.evaluate(stream=data_stream, model = CLF, model_names  = ['HT','HAT'])
        print("********************************")
        print("Results for "+ str(os.path.basename(x)))
        print("********************************")
    except:
        print("Error......!!")


