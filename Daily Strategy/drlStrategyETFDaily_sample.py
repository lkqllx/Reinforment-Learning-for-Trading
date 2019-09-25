#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:48:03 2018

@author: hsc
"""

import pandas as pd
import numpy as np

ETFs = ["159920.SZ","510900.SH","518880.SH","511010.SH","159915.SZ","510050.SH","510300.SH","510500.SH"]

modelInitialLookbackWindowSize = 400
lookbackNumWindowsForSignals = 20

outputDir = "C:\\ETFDaily\\"
featureDir = "C:\\ETFDailyFeatures\\"

readInCols = ["Date","VWAP","todayOpen","zt","zt0","zt1","zt2","zt3","zt4"]

features = {}

for ticker in ETFs:
    features[ticker] = pd.read_csv(featureDir + "features_" + ticker + ".csv", usecols = readInCols)

#Build models for individual ETFs with tensorflow
##Set the right device for computation
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Only use GPU 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
import pandas as pd

##Check if GPU is used correctly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

## to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

pastDays = modelInitialLookbackWindowSize

## iterate through all ETF tickers
for ticker in ETFs:
    featureData = features[ticker]
    #Iterate through days
    dates=featureData.Date
    date_index=np.unique(dates)
    date_num=date_index.size
    
    # hyper parameters
    n_stepsArray = [20, 200]  # T: length of recurrent cells
    n_inputs = 5  # number of input features
    n_hidden1 = 128
    n_hidden2 = 128
    n_hidden3 = 128
    n_hidden4 = 16
    n_neurons = 1  # number of recurrent neurons
    n_outputs = 1  # output dimension
    
    n_epochs = 200
    batch_size = 200
            
    c = .0002  # transaction fee level
    learning_rate = 0.001
    
    for n_steps in n_stepsArray:
        cols = ['ticker','date','delta','nextCloseToOpen','nextDayOpen','TR','SR','nDCtoCPnL','nDCtoCCumuPnL','nDCtoCPnLMinusTC','nDCtoCCumuPnLMinusTC']
        outputFile = pd.DataFrame(columns=cols)
    
        print("date_num=" + str(date_num))
        print("pastDays=" + str(pastDays))
        cumuPnL = 0
        cumuPnLMinusTC = 0
        for i in range(date_num-pastDays-1):
            trainDates=[date_index[i]]
            for j in range(i+1,i+pastDays):
                trainDates.append(date_index[j])        
        
            dataPd=featureData[featureData.Date.isin(trainDates)]
            dataNextD=featureData[featureData.Date==date_index[i+pastDays]]
        
            #Prepare data
            startIdx = 4
            F_train_all = dataPd.iloc[:,startIdx:(n_inputs+startIdx)].values
            z_train_all = dataPd.zt.values
            p_train_all = dataPd.todayOpen.values
            
            F_train = F_train_all[:int(len(dataPd)/n_steps)*n_steps, :]
            z_train = z_train_all[:int(len(dataPd)/n_steps)*n_steps]
            p_train = p_train_all[:int(len(dataPd)/n_steps)*n_steps]
            
            # Build computation graph
            reset_graph()
            
            f = tf.placeholder(tf.float32, shape=(None, n_inputs), name="input")
            z = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name="z")
            p = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name="p")
            #seq_length = tf.placeholder(tf.int32, [None], name="seq_length")
            
            with tf.name_scope("DNN"):
                hidden1 = tf.layers.dense(f, n_hidden1, activation=tf.nn.selu, name="hidden1")
                hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.selu, name="hidden2")
                hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.selu, name="hidden3")
                hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.selu, name="hidden4")
            
            F = tf.reshape(hidden4, [-1, n_steps, n_hidden4])
            
            cell = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=None, name="rnn"), output_size=n_outputs)
            
            deltaTemp, states = tf.nn.dynamic_rnn(cell, F, dtype=tf.float32)
            delta = tf.nn.relu(deltaTemp, name="deltaCalc")
            
            R = tf.pad(delta[:, :(n_steps-1), :] * z[:, 1:(n_steps), :] - 
                        tf.abs(delta[:, 1:n_steps, :] - delta[:, :(n_steps-1), :]) * p[:, 1:(n_steps)] * c,
                             paddings=[[0, 0], 
                                       [1, 0],
                                       [0, 0]])
            
            U = tf.reduce_mean(tf.reduce_sum(R, axis=1))
            
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            
            TR = tf.reduce_sum(R) # total return
            SR = tf.reduce_mean(R)/(tf.sqrt(tf.nn.moments(tf.reshape(R, [-1]), axes=0)[1]) + 1e-10)*np.sqrt(252) # Sharpe Ratio
            
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            
            #Train and test the model
            with tf.Session() as sess:
                #Train the model
                init.run()
                for epoch in range(n_epochs):
                    for iteration in range(F_train.shape[0] // batch_size):
                        f_batch = F_train[iteration*batch_size:(iteration+1)*batch_size, :]
                        z_batch = z_train[iteration*batch_size:(iteration+1)*batch_size]
                        p_batch = p_train[iteration*batch_size:(iteration+1)*batch_size]
                        z_batch = z_batch.reshape((-1, n_steps, n_outputs))
                        p_batch = p_batch.reshape((-1, n_steps, n_outputs))
                        sess.run(optimizer.apply_gradients(optimizer.compute_gradients(-U)), 
                                                             feed_dict={f: f_batch, z: z_batch, p: p_batch})

                #Test the model                
                z_input = z_train.reshape((-1, n_steps, n_outputs))
                p_input = p_train.reshape((-1, n_steps, n_outputs))
    
                delta_sanityCheck = delta.eval(feed_dict={f: F_train, z: z_input, p:p_input}).flatten()
                U_sanityCheck = U.eval(feed_dict={f: F_train, z: z_input, p: p_input}).flatten()
                TR_sanityCheck = TR.eval(feed_dict={f: F_train, z: z_input, p: p_input}).flatten()
                SR_sanityCheck = SR.eval(feed_dict={f: F_train, z: z_input, p: p_input}).flatten()
                nextDayPnL = delta_sanityCheck.flatten()[-1] * dataNextD['zt'].values[-1]
                if i == 0: #first day
                    nextDayPnLMinusTC = nextDayPnL - np.abs(delta_sanityCheck.flatten()[-1]) * c
                else:
                    nextDayPnLMinusTC = nextDayPnL - np.abs(delta_sanityCheck.flatten()[-1] - delta_sanityCheck.flatten()[-2]) * c
                cumuPnL = cumuPnL + nextDayPnL
                cumuPnLMinusTC = cumuPnLMinusTC + nextDayPnLMinusTC

                resultForThisDay = [ticker, str(date_index[i+pastDays]), delta_sanityCheck.flatten()[-1], dataNextD['zt'].values[-1], dataNextD['todayOpen'].values[-1], TR_sanityCheck[-1], SR_sanityCheck[-1],\
                                    nextDayPnL, cumuPnL, nextDayPnLMinusTC, cumuPnLMinusTC]
                outputFile = outputFile.append(pd.DataFrame(data=[resultForThisDay], columns=cols))
                print(resultForThisDay)
        fileName = ticker + "_" + "n_steps=" + str(n_steps) + ".csv"
        outputFile.to_csv(fileName)