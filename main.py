from config import args
import numpy as np
from models import logistic,trees,svm,mlp
from data.dataloader import data
from predict import accuracy, metrics
from preprocess import Preprocessing
from data import dataloader
from models import adaBoostClassifier, Bagging, voting
import warnings
warnings.filterwarnings('ignore')


path = args.path

df = dataloader.data(path)


x_train,y_train, x_test,y_test =  Preprocessing(df).split(df,percent=args.percent)
x_under, y_under = Preprocessing(df).under_sampler(x_train, y_train)
x_over, y_over = Preprocessing(df).over_sampler(x_train, y_train)

'''Combining the dataset in order to get the training data.'''
over_train = x_over.join(y_over) 
under_train = x_under.join(y_under)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', help='This is the name of the model', required=True, type=str)
# parser.add_argument('-n', '--num_epochs', help='This is the number of epochs', required=True, type=int)

mains_args= vars(parser.parse_args())

# num_epochs= mains_args['num_epochs']



def call_models(adaboost=False, bagging=False,vote=False):
    models = [logistic,trees,svm,mlp]

    bag = Bagging()

    if adaboost:
        '''ada'''
        # ada_under = adaBoostClassifier(n_estimators=args.n_estimators, random_state=42)
        # ada_under.fit(x_under, y_under)
        # ada_under.predict(x_test)

        ada_over = adaBoostClassifier(n_estimators=args.n_estimators, random_state=42)
        ada_over.fit(x_over, y_over)
        y_pred =ada_over.predict(x_test)
        acc = accuracy(y_test,y_pred)
        metrics(y_test,y_pred)
    
    elif bagging:
        '''bagging'''
        y_pred = bag.bagging(over_train,x_test,y_test,n_trees=5)
        acc = accuracy(y_test,y_pred)
        metrics(y_test,y_pred)

    elif vote:
        '''voting'''
        y_pred = voting(x_over, y_over,x_test,y_test,models)
        acc = accuracy(y_test,y_pred)
        metrics(y_test,y_pred)

if mains_args['model_name'].lower()=='adaboost':
    model= call_models(adaboost=True)

elif mains_args['model_name'].lower()=='voting':
    model= call_models(vote=True)

elif mains_args['model_name'].lower()=='bagging':
    model= call_models(bagging=True)
