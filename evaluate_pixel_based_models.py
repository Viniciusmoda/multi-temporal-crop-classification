
# Classifiers used in Zhong paper (apart from CNN and LSTM-based methods)
#
######

import numpy as np
#np.random.seed(100)
import os, glob
import pandas as pd
from random import shuffle
import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical
#from visualize import visualize_time_series
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pandas as pd
from pathlib import Path
import sys
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import xgboost

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
    curr_ts_data = curr_ts['NDVI'].values.reshape(1, -1)
    curr_ts_data = replace_nans(curr_ts_data)
    curr_ts_data = StandardScaler().fit_transform(curr_ts_data)
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
    allY = np.array(allY)  
    return [allX, allY]


def train_val_test(classifier, trainX, trainY, valX, valY, testX, testY):
    classifier.fit(trainX, trainY)
    predictions = classifier.predict(testX)
    print(predictions)

if __name__ =="__main__":
    trainX, trainY = prepare_dataset('/home/kgadira/crop-classification/1_data/filtered-extracts-subset-ts', 'train')
    valX, valY = prepare_dataset('/home/kgadira/crop-classification/1_data/filtered-extracts-subset-ts', 'val')
    testX, testY = prepare_dataset('/home/kgadira/crop-classification/1_data/filtered-extracts-subset-ts', 'test')
    print(trainX.shape, trainY.shape)
    classifier = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=[1]*23))
    train_val_test(classifier, trainX, trainY, None, None, testX, None)
