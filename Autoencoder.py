#!/usr/bin/env python3

import os
import sys
import math

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import scipy.io
from sklearn.utils.extmath import randomized_svd
import buildmodel
import time

# Class for printing history
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      newfile=open('Out.txt','a+')
      newfile.write('Epoch: '+str(epoch)+'   ')
      newfile.write('Loss: '+str(logs['loss'])+'\n')
      newfile.close()

###############################################################################
# Plotting Functions
###############################################################################
# For plotting history
def plot_history(histshift):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.semilogy(histshift['epoch'], histshift['loss'],
       label='Train Loss')
    plt.semilogy(histshift['epoch'], histshift['val_loss'],
       label = 'Val Loss')
    plt.legend()
    #plt.ylim([10**-6,10**-1])


def plot_parity(utt_test,test_predictions,KSmax=3.5):
    plt.plot(utt_test.flatten(), test_predictions.flatten(),'o',color='black',markersize=.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([-KSmax,KSmax])
    plt.ylim([-KSmax,KSmax])
    plt.plot([-100, 100], [-100, 100])

def plot_hist(error,title,bins=40,ymax=50,xmax=.05):
    plt.hist(error, bins = bins,density=True)
    plt.xlim([-xmax,xmax])
    plt.ylim([0,ymax])
    plt.xlabel("Prediction Error")
    variance=np.mean(error**2)
    plt.title('MSE='+str(round(variance,7)))
    plt.ylabel(title+' PDF')
    
    return variance

def scheduler(epoch, lr):
    if epoch < 500:
        return .001
    else:
        return .0001

###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':
    # Load data
    u_train=pickle.load(open('Data_Train.p','rb'))
    u_test=pickle.load(open('Data_Test.p','rb'))
    [M,N]=u_train.shape

    u_mean=np.mean(u_train,axis=0)
    u_std=np.std(u_train,axis=0)
    u_train=(u_train-u_mean[np.newaxis,:])/u_std[np.newaxis,:]
    u_test=(u_test-u_mean[np.newaxis,:])/u_std[np.newaxis,:]

    # Perform SVD on the training data
    U,S,VT=randomized_svd(u_train.transpose(), n_components=N)
    pickle.dump([u_mean,u_std,U],open('PCA.p','wb'))

    # Do a full change of basis
    a_train=u_train@ U
    a_test=u_test@ U

    # Store errors for figure
    MSEsig=[]
    MSEpca=[]

    # For timing training
    timer=time.time()
    # Loop over latent dimensions
    for trunc in range(5,11):
        ###########################################################################
        # Autoencoder
        ###########################################################################
        # Build model
        model,EPOCHS,optimizer=buildmodel.buildmodel(N,trunc)

        # Compile model
        model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])
        
        # Train model 
        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        history = model.fit(a_train, a_train,epochs=EPOCHS, validation_data=(a_test[:1000:10,:],a_test[:1000:10,:]), verbose=0,callbacks=[PrintDot(),callback])

        # Save history for comparing models
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        ###########################################################################
        # Plot Results
        ###########################################################################
        
        # Plot MAE and MSE vs epochs
        plot_history(hist)
        plt.tight_layout()
        plt.savefig(open('Error'+str(trunc)+'.png','wb'))
        plt.close()

        pred_a=model.predict(a_test)    
        # Real Space NN prediction
        pred_u=pred_a@U.transpose()
        # Real Space PCA prediction
        PCA_u=u_test@ U[:,:trunc]@ U[:,:trunc].transpose()

        # Plot side by side histograms
        plt.figure()
        # Plot side by side histograms
        plt.subplot(1,2,1)
        error = u_test.flatten() - pred_u.flatten()
        MSE=plot_hist(error,'NN')
        plt.subplot(1,2,2)
        error = u_test.flatten() - PCA_u.flatten()
        MSE_PCA=plot_hist(error,'PCA')
        plt.tight_layout()
        plt.savefig(open('Statistics'+str(trunc)+'.png','wb'))
        plt.close()
        
        # Save models
        model.save_weights('model'+str(trunc)+'.h5')

        # Save MSE
        pickle.dump([MSE,MSE_PCA],open('MSE'+str(trunc)+'.p','wb'))
        MSEsig.append(MSE)
        MSEpca.append(MSE_PCA)
    
    # Output the time it took to train
    newfile=open('Out.txt','a+')
    newfile.write('Time: '+str(time.time()-timer))
    newfile.close()

    # Plot the Errors for comparison
    HL=np.arange(5,11)
    plt.figure()
    plt.semilogy(HL, MSEsig,'x',label='Sig')
    plt.semilogy(HL,MSEpca,'s',label='PCA')
    plt.legend()
    plt.xlabel(r'$d_h$')
    plt.ylabel('MSE')
    plt.savefig(open('Comparison.png','wb'))

 
