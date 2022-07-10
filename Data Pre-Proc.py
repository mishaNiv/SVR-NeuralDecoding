#importing basic packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
import sys
import pandas as pd

#loading .mat data
folder='/Users/mnivota/OneDrive/College Prep/Polygence/MatLab/'
data = io.loadmat(folder + 'Raw_Neural_Data.mat')

#separating different parts of data into vars
units = data['unit'] #load units (matrix with timee, labels, etc. for 143 neurons)
s_times = units['times'] #load spike times of each neuron
directions = data['direction'] #load directions (output) for each trial
directions = np.array(directions, dtype = 'float64') #setting directions as array
g_times = data['go'] #loading go times for each trial

#creating and setting initial vars and arrays
rows = 158 #total trials (rows of final matrix)
cols = 143 #total neurons (columns of final matrix)

#creating final matrix, initialized with 0s
neural_data = [[0 for _ in range(cols)] for _ in range(rows)] 

start = 0 #oo time of current trial (start time)
end = 0 #go time of next trial
    #(start time of next trial AKA end time of current trial)

#separating continuous individual neural data by trials and adding spikes per trial
    #then inputing total spikes into matrix of trials by neurons
for cur_go in range(rows):
    start = g_times[cur_go][0] #setting start time to current trials' go time
    if (cur_go != rows-1): 
        end = g_times[cur_go+1][0] #setting end time to next trial's go time 
    else:
        end = 10000 #for the last trial, setting end to large number (no next trial, so no end time)

    spikes = [0 for i in range(cols)] #array of spike times, initialized to 0s

    for cur_neuron in range(cols): #separating neurons
        temp = s_times[0][cur_neuron] #temporary array of spike times for each neuron
        
        for cur_sp in range(temp.size): 
            if temp[cur_sp] >= start: #if spike time is over start time
                if temp[cur_sp] < end: #if spike time in under end time
                    spikes[cur_neuron] += 1 #adding 1 to spike count for current neuron
                else:
                    break #otherwise, we've gone past end of next trial and can stop

    #combining and defining arrays 
    spikes = np.array(spikes, dtype = 'float64') #transforming spikes to array
    neural_data[cur_go] = spikes #setting current row of final matrix (current trial) as spikes

#transforming final matrix from lists to arrays
neural_data = np.array(neural_data, dtype = 'float64')

#transforming and putting .mat data into .pickle data for model
data_folder = '/Users/mnivota/OneDrive/College Prep/Polygence/MatLab/' #file location to-be

with open(data_folder + 'Lab6_Dat.pickle','wb') as f:
    pickle.dump([neural_data,directions],f) #dumping data as .pickle file
