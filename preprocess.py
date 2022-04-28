

import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class Preprocessing():

    def __init__(self,data):
        self.data=data
        '''split function'''

    def split(self,data,percent):

        np.random.seed(42)
        perm = np.random.permutation(data.index)
        n = len(data)

        train_size = int(percent* n)
        train_data = data.iloc[:perm[train_size]:]
        test_data  = data.iloc[perm[train_size]:]
        x_train = train_data.iloc[: , :-1 ]
        y_train =  train_data.iloc[ : , -1 ]

        x_test  = test_data.iloc[: , :-1 ]
        y_test  = test_data.iloc[ : , -1 ]

        return x_train,y_train, x_test,y_test



    ''' for undersampling'''
    def under_sampler(self,X, y):
        undersample = RandomUnderSampler(sampling_strategy='majority') #instantiate the model
        X_under, y_under= undersample.fit_resample(X, y)   # fit and apply the transform

        return X_under,y_under

    ''' for oversampling'''
    def over_sampler(self,X, y):
        oversample = RandomOverSampler(sampling_strategy='minority') #instantiate the model
        X_over,y_over= oversample.fit_resample(X, y)   # fit and apply the transform
        
        return X_over,y_over