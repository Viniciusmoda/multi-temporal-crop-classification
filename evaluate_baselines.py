
# Classifiers used in Zhong paper (apart from CNN and LSTM-based methods)
#
######

import numpy as np
#np.random.seed(100)
import os, glob
import pandas as pd
from random import shuffle
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pandas as pd
from pathlib import Path
import sys
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
from sklearn.svm import SVC 

def replace_nans(data):
    """
    Replace any nans with nearest values
    Source: https://stackoverflow.com/questions/9537543/replace-nans-in-numpy-array-with-closest-non-nan-value
    """
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    return(data)


def preprocess_data(ts_path):
    curr_class = int(os.path.basename(os.path.dirname(ts_path)))
    curr_ts = pd.read_csv(ts_path)
    if curr_ts.empty:
        print("{} seems to be empty: pls check ".format(ts_path))
        sys.exit(0)
    curr_ts_data = curr_ts['NDVI'].values
    curr_ts_data = curr_ts_data[:, np.newaxis]
    curr_ts_data = replace_nans(curr_ts_data)
    curr_ts_data = StandardScaler().fit_transform(curr_ts_data)
    curr_ts_data = curr_ts_data.T
    return [curr_ts_data, curr_class]        


def prepare_dataset(input_path, mode="train", num_classes =6, do_shuffle=True):	
    """
    Extracts the time series from dataset, interpolates missing data and returns a numpy array for training/validation/testing

    """
    data_path = os.path.join(input_path, mode)
    all_ts = glob.glob(os.path.join(data_path, "**/*.csv"))
    assert len(all_ts) > 0, "Something wrong with input path, please check"
    print("Found {} files for {}".format(len(all_ts), mode))
    allX = None
    allY = []
    if do_shuffle:
        shuffle(all_ts)
    for ts_path in all_ts: 
        ts = []
        labels = []       
        tmp_X, tmp_Y = preprocess_data(ts_path=ts_path) 
        if allX is None:
            allX = tmp_X.copy()
            allY.append(tmp_Y)
        else:   
            allX = np.vstack((allX, tmp_X))
            allY.append(tmp_Y)
    print(allX.shape)
    allY = np.array(allY)  
    return [allX, allY]


def train_predict(clf_key, classifier, param_grid, trainX, trainY, valX, valY, testX, testY):
    all_tr_val_X = np.vstack((trainX, valX))
    all_tr_val_Y = np.hstack((trainY, valY))
    print(all_tr_val_X.shape, all_tr_val_Y.shape)
    fold_meta = np.zeros(all_tr_val_X.shape[0])
    fold_meta[0:trainX.shape[0]] = -1
    cv = PredefinedSplit(test_fold=fold_meta)     
    gcv = GridSearchCV(estimator = classifier, param_grid = param_grid, cv=cv, verbose=0, n_jobs=-1, scoring='accuracy')
    gcv.fit(all_tr_val_X, all_tr_val_Y)
    predictions = gcv.predict(testX)
    cm = confusion_matrix(testY, predictions)
    classes_lst = ['Corn', 'Cotton', 'Soy', 'Spring Wheat', 'Winter Wheat', 'Barley']
    creport = classification_report(y_true = testY, y_pred=predictions, target_names = classes_lst, digits = 4, output_dict = True)
    creport_df = pd.DataFrame(creport).transpose()
    acc = accuracy_score(testY, predictions)
    print(creport)
    kappa_score = cohen_kappa_score(testY, predictions)
    print(f'Classifier is {clf_key}, best params are {gcv.best_params_}, Accuracy is {acc}, Kappa Score is {kappa_score}\n confusion matrix is {cm}\n clf report is {creport}')

if __name__ =="__main__":
    trainX, trainY = prepare_dataset('./1_data/filtered-extracts-subset-ts', 'train')
    valX, valY = prepare_dataset('./1_data/filtered-extracts-subset-ts', 'val')
    testX, testY = prepare_dataset('./1_data/filtered-extracts-subset-ts', 'test')
    print(np.unique(trainY), np.unique(valY), np.unique(testY))
    print(trainX.shape, trainY.shape)
    print(trainX[523, :])
    # these are parameters specified in the paper
    classifiers_grid = {
                        'rf':{'n_estimators': [120,300,500,800,1200],
                        'max_depth': [5,8,15,25,30,None],
                        'min_samples_split': [2,5,10,15,100],
                        'min_samples_leaf': [1,2,5,10],
                        'max_features': ['log2', 'sqrt', None]
                        },
                        'xgb':{
                            'learning_rate':[0.01, 0.015, 0.025, 0.05, 0.1],
                            'gamma':[0.05,0.1,0.3,0.5,0.7,0.9,1],
                            'max_depth':[5,7,9, 12,15,17,25],
                            'min_child_weight':[1,3,5,7],
                            'subsample':[0.6, 0.7, 0.8, 0.9, 1],
                            'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1],
                            'reg_lambda':[0.01, 0.1, 1],
                            'reg_alpha':[0, 0.1, 0.5, 1]
                        },
                        'svc':{
                            'C': [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000],
                            'gamma':[0.1, 1, 2, 10, 'auto']
                        }

                        }
    
    classifiers = {
                    "rf": RandomForestClassifier(),
                    "xgb": xgb.XGBClassifier(),
                    "svc": SVC()
                }
    for clf_key in ['rf', 'xgb', 'svc']:
        classifier = classifiers[clf_key]
        param_grid = classifiers_grid[clf_key]
        train_predict(clf_key, classifier, param_grid, trainX, trainY, valX, valY, testX, testY)
