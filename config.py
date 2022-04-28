import argparse



args = argparse.Namespace(
    path = './data/breast-cancer-wisconsin.csv',
    n_estimators = 4, #the number of trees to be specified
    max_depth = 1, 
    max_leaf_nodes = 3, 
    percent = 0.75,
    ratio=0.4
   
)