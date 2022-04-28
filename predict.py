import numpy as np


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def accuracy(y_test,ypred):
  ''' we compute the accuracy of our predictions from the bagging method.'''
#   # out=0
#   n=len(y_test)
#   out = np.sum([1 if y_test[i]==ypred[i] else 0 for i in range(len(y_test))])
  return np.mean(ypred==y_test)

def metrics(y_test,pred):

    print(f'{classification_report(y_test, pred)}')
    print(f'{confusion_matrix(y_test, pred)}') 


