import numpy as np
import pandas as pd

from config import args
from data.dataloader import data
from random import randrange
from preprocess import Preprocessing

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


logistic = LogisticRegression() # instantiating the model
svm = SVC()
trees = DecisionTreeClassifier(max_leaf_nodes=3, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)

'''This is voting ensemble method'''

def voting(x_over, y_over,x_test,y_test,models):
  predictions = np.zeros((len(y_test),len(models) ))
  for m  in range(len(models)):
    model=models[m].fit(x_over, y_over)
    predictions[:,m] = model.predict(x_test)
  
  ''' Here, we compute the majority voting.'''
  predictions = pd.DataFrame(predictions)  
  final_predictions = [max(set(list(predictions.values[i])),key=list(predictions.values[i]).count) for i in range(len(predictions))]
  return final_predictions


'''This is bagging ensemble method'''
class Bagging():

    def __init__(self):
        
      '''This function outputs the different samples from the same training dataset. Note : the sample is a random sample with replacement.'''
    def bootstrap_sample(self,original_dataset, ratio=args.ratio): 
        sub_dataset = []
        n_elt = round(len(original_dataset)*ratio) 
        for i in range(n_elt):
        #   '''the index outputs a random index of a datapoint from training dataset.'''
          
            index = randrange(len(original_dataset)) 

            sub_dataset.append(original_dataset.values[index])
        return sub_dataset

    def build_tree(self,x_train,y_train):
        tree = DecisionTreeClassifier(max_leaf_nodes=3, max_depth=args.max_depth,random_state=42)
        # tree = SVC()
        # tree = LogisticRegression()
        # tree = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
        tree.fit(x_train,y_train)
        return tree 

    ''' Make a prediction with a list of bagged trees'''
    def bagging_predict(self,tree,x):
        pred = tree.predict(x)
        return pred

    def bagging(self,over_train,x_test,y,n_trees):
        predictions = np.zeros((len(y),n_trees))
        trees = []
        for i in range(n_trees):
            sample = self.bootstrap_sample(over_train,ratio=args.ratio)
            sample = pd.DataFrame(sample)
            tree = self.build_tree(sample.iloc[:,:-1],sample.iloc[:,-1])
            # print(tree)
            predictions[:,i] = self.bagging_predict(tree,x_test)

            ''' majority voting'''
        predictions = pd.DataFrame(predictions)
        # print(predictions)
        final_predictions = [max(set(list(predictions.values[i])),key=list(predictions.values[i]).count) for i in range(len(predictions))]
        return final_predictions

        
'''This is boosting ensemble method'''
class adaBoostClassifier():

  def __init__(self, n_estimators = args.n_estimators, max_depth = args.max_depth, max_leaf_nodes= args.max_leaf_nodes, random_state=None):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.max_leaf_nodes = max_leaf_nodes
    self.random_state = random_state
    self.sample_weight = np.array([])
    self.weight_model = dict()
    self.model = dict()
    self.features = dict()

  def fit(self, x, y):

    # weigth initialization
    x_length = len(x)
    self.sample_weight = np.ones(x_length)/x_length


    np.random.seed(seed=9999)
    for i in range(self.n_estimators) :

      # randomly select a sample
      rand_index = np.random.randint(low=0, high=x.shape[0], size=round(x.shape[0]*0.8))
      rand_column = np.random.randint(low=0, high=x.shape[1], size=round(x.shape[1]*0.8))

      x_samp = x.iloc[rand_index, :]
      x_samp = x_samp.iloc[:, rand_column]
      y_samp = y.iloc[rand_index]
      weight_samp = self.sample_weight[rand_index]

      # Initialize the model (decision tree)
      # decision_tree_model = DecisionTreeClassifier(max_leaf_nodes=self.max_leaf_nodes, random_state=self.random_state, max_depth=self.max_depth)

      # Initialize the model (svm)
      # decision_tree_model = SVC()

      # Initialize the model (svm)
      # decision_tree_model = LogisticRegression()

      # Initialize the model (svm)
      decision_tree_model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)

      # fit the model
      self.model[i] = decision_tree_model.fit(x_samp, y_samp)
      # self.model[i] = decision_tree_model.fit(x_samp, y_samp, sample_weight=weight_samp)
      self.features[i] = rand_column

      # Prediction 
      y_pred = self.model[i].predict(x_samp)

      # Compute the total error
      total_error = np.sum((y_pred!=y_samp) * weight_samp)

      # Compute the weight of the model
      self.weight_model[i] = (1/2) * np.log((1-total_error)/(total_error))

      # updating the weight
      self.sample_weight[rand_index] = (y_pred!=y_samp) * self.sample_weight[rand_index] * np.exp(self.weight_model[i]) + (y_pred==y_samp) * self.sample_weight[rand_index] * np.exp(-self.weight_model[i])
  
      # Normalize the weight  
      self.sample_weight /= np.sum(self.sample_weight)


  def predict(self, x):

    pred = np.zeros(x.shape[0])
    for i in range(self.n_estimators):
      pred += self.model[i].predict(x.iloc[:, self.features[i]]) * self.weight_model[i]

    pred = pred/self.n_estimators
    pred = np.where(pred <= 3, 2, 4)

    return pred





