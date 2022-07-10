#import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
import sys

#import decoder and function for covariate matrix
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history
from Neural_Decoding.decoders import SVRDecoder

#import metrics
from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho
import pandas as pd

#setting var as location of data
data_folder = '/Users/mnivota/OneDrive/College Prep/Polygence/MatLab/' 

#opening and reading in .pickle data file
with open(data_folder + 'Lab6_Dat.pickle','rb') as f:
    neural_data, directions = pickle.load(f,encoding='latin1')

bins_before=0 #ignoring previous trial data in decoding
bins_current=1 #using current trial data for decoding
bins_after=0 #ignoring following trial data in decoding

#removing neurons with too few spikes
nd_sum=np.nansum(neural_data,axis=0) #total number of spikes of each neuron
rmv_nrn=np.where(nd_sum<1) #find neurons who have less than 1 spikes total
neural_data=np.delete(neural_data,rmv_nrn,1) #remove those neurons

#creating covariate matrix with trial spike data
X=get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)
X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))

#seting decoding output
y=directions

#remove time bins with no output (direction)
rmv_time=np.where(np.isnan(y[:,0])) #find time bins with no output
X=np.delete(X,rmv_time,0) #remove those time bins from X
X_flat=np.delete(X_flat,rmv_time,0) #remove those time bins from X_flat
y=np.delete(y,rmv_time,0) #remove those time bins from y

#setting how much data is used for training, testing, and validation
training_range=[0, 0.60]
valid_range=[0, 0.60]
testing_range=[0.60, 1.0]

num_examples=X.shape[0]

#setting training, testing, and validation data
training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)
valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

#splitting training into x, flat, and y
X_train=X[training_set,:,:]
X_flat_train=X_flat[training_set,:]
y_train=y[training_set,:]

#splitting testing into x, flat, and y
X_test=X[testing_set,:,:]
X_flat_test=X_flat[testing_set,:]
y_test=y[testing_set,:]

#splitting validation into x, flat, and y
X_valid=X[valid_set,:,:]
X_flat_valid=X_flat[valid_set,:]
y_valid=y[valid_set,:]

#finding z-scores for x
X_train_mean=np.nanmean(X_train,axis=0)
X_train_std=np.nanstd(X_train,axis=0)
X_train=(X_train-X_train_mean)/X_train_std
X_test=(X_test-X_train_mean)/X_train_std
X_valid=(X_valid-X_train_mean)/X_train_std

#finding z-scores for x_flat
X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
X_flat_train_std=np.nanstd(X_flat_train,axis=0)
X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
X_flat_valid=(X_flat_valid-X_flat_train_mean)/X_flat_train_std

#normalizing output by centering it around 0
y_train_mean=np.mean(y_train,axis=0)
y_train=y_train-y_train_mean
y_test=y_test-y_train_mean
y_valid=y_valid-y_train_mean

#finding z-scores for y 
y_train_std=np.nanstd(y_train,axis=0)
y_zscore_train=y_train/y_train_std
y_zscore_test=y_test/y_train_std
y_zscore_valid=y_valid/y_train_std

#declaring model
model_svr=SVRDecoder(C=5, max_iter=4000)

#fitting model
model_svr.fit(X_flat_train,y_zscore_train)

#geting predictions
y_zscore_valid_predicted_svr=model_svr.predict(X_flat_valid)

#geting metric of fit
R2s_svr=get_R2(y_zscore_valid,y_zscore_valid_predicted_svr)
print('R2s:', R2s_svr)



#making plots
    #not necessary
    #maybe confusion matrix??
    #can present r^2 with more detail
        #find which neurons weren't predicted accuracy
        #compare with Omar's original code (maybe it's a problem with the data)
        