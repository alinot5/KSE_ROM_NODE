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
import seaborn as sns
from matplotlib.colors import ListedColormap

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

import scipy.io
from sklearn.utils.extmath import randomized_svd
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
import torch.optim as optim
import buildmodel_large

# This is the class that contains the NN that estimates the RHS
class ODEFunc(nn.Module):
    def __init__(self,trunc):
        super(ODEFunc, self).__init__()
        # Change the NN architecture here
        self.net = nn.Sequential(
            nn.Linear(trunc, 200),
            nn.Sigmoid(),
            nn.Linear(200, 200),
            nn.Sigmoid(),
            nn.Linear(200,200),
            nn.Sigmoid(),
            nn.Linear(200, trunc),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        y=torch.tensor(y)
        y=y.type(torch.FloatTensor)
        return self.net(y).detach().numpy()

###############################################################################
# Plotting
###############################################################################
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def plot_difference(t_true,u_true,t_pred,u_pred):
    N=64
    colors=sns.diverging_palette(240, 10, n=9,as_cmap=True)
    colors2=sns.diverging_palette(240, 10, n=41)
    colors2 = ListedColormap(colors2[20:])
    
    width=8.6
    plot_size=cm2inch(width,width*4/3)
    
    fig, axs = plt.subplots(4, 1, sharex='col', sharey='row',figsize=plot_size,dpi=500)
    (ax1),(ax2),(ax3),(ax4) = axs
    im=ax1.pcolormesh(t_true, np.linspace(-11,11,N), u_true, shading='gouraud', cmap=colors,vmin=-3,vmax=3)
    im2=ax2.pcolormesh(t_pred, np.linspace(-11,11,N), u_pred, shading='gouraud', cmap=colors,vmin=-3,vmax=3)
    im3=ax3.pcolormesh(t_pred, np.linspace(-11,11,N), np.abs(u_pred-u_true), shading='gouraud', cmap=colors2,vmin=0,vmax=3)
    im4=ax4.plot(t_pred,np.linalg.norm(u_pred-u_true,axis=0))
    
    ax1.set_xlim([0,100]);ax2.set_xlim([0,100]);ax3.set_xlim([0,100]);ax4.set_xlim([0,100])
    ax1.set(ylabel=r'$x$')
    ax2.set(ylabel=r'$x$')  
    ax3.set(ylabel=r'$x$')
    ax4.set(ylabel=r'$||u-\tilde{u}||$')
    ax4.set(xlabel=r'$t$')
    
    cax = fig.add_axes([.93, 0.795, 0.015, 0.165]) #left bottom width height
    cb=fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label(r'$u$')
    cax2 = fig.add_axes([.93, 0.573, 0.015, 0.165]) #left bottom width height
    cb2=fig.colorbar(im2, cax=cax2, orientation='vertical')
    cb2.set_label(r'$\tilde{u}$')
    cax3 = fig.add_axes([.93, 0.352, 0.015, 0.165]) #left bottom width height
    cb3=fig.colorbar(im3, cax=cax3, orientation='vertical')
    cb3.set_label(r'$|u-\tilde{u}|$')
    
    for ax in axs.flat:
        ax.label_outer()

###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':
    # Load data
    [u_test,ttest]=pickle.load(open('u66.p','rb'))
    [M,N]=u_test.shape

    trunc=28
    model,_,_=buildmodel_large.buildmodel(N,trunc)
    model.load_weights('model66.h5')
    encode = Model(inputs=model.input,outputs=model.get_layer('Dense_In3').output)

    [u_mean,u_std,U]=pickle.load(open('PCA66.p','rb'))
    a_test=(u_test-u_mean[np.newaxis,:])/u_std[np.newaxis,:]@ U
    h=encode.predict(a_test)

    decode=buildmodel_large.decode(N,trunc)
    for i in range(3):
        ly='Dense_Out'+str(i+1)
        decode.get_layer(ly).set_weights(model.get_layer(ly).get_weights())

    ###########################################################################
    # Evolve dh_dt
    ###########################################################################
    # Load ODE
    g=torch.load('model66.pt')

    # Evolve data
    sol = solve_ivp(g.forward, [0, ttest[-1]], h[0,:],t_eval=ttest)
    # Decode data
    aNN=decode.predict(sol.y.T)
    # Move out of PCA basis and fix normalization
    utemp=aNN@U.transpose()*u_std[np.newaxis,:]+u_mean[np.newaxis,:]

    # Plot trajectories
    plot_difference(ttest,u_test.T,ttest,utemp.T,)
    plt.tight_layout()
    plt.savefig('Pred66.png',bbox_inches="tight")









