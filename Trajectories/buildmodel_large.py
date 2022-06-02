#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,Lambda
import tensorflow.keras.backend as K
from functools import reduce
import os

#Note: All operations must go into Lambda layers
# Dense layer equivariance in main function
def buildmodel(N,trunc):
    K.set_floatx('float64')
    #Parameters
    optimizer = tf.keras.optimizers.Adam()
    EPOCHS=1000
    hiddenin=[500,500,trunc]
    actin=['sigmoid','sigmoid',None]
    hiddenout=[500,500,N]
    actout=['sigmoid','sigmoid',None]

    # Input size
    main_input = layers.Input(shape=(N,), name='main_input')

    ########################################################################################
    # Encoder
    ########################################################################################
    # Nonlinear dim increase
    encode=main_input
    for i in range(len(hiddenin)):
        encode=layers.Dense(hiddenin[i],activation=actin[i],name='Dense_In'+str(i+1))(encode)

    hidden=encode

    ########################################################################################
    # Decoder
    ########################################################################################
    # Nonlinear dim increase
    decode=hidden
    for i in range(len(hiddenout)):
        decode=layers.Dense(hiddenout[i],activation=actout[i],name='Dense_Out'+str(i+1))(decode)

    main_output=decode

    # Build model
    model=Model(inputs=main_input,outputs=main_output)
    
    return model,EPOCHS,optimizer

# Dense layer equivariance in main function
def encode(N,trunc):
    K.set_floatx('float64')
    #Parameters
    hiddenin=[500,500,trunc]
    actin=['sigmoid','sigmoid',None]

    # Input size
    main_input = layers.Input(shape=(N,), name='main_input')

    ########################################################################################
    # Encoder
    ########################################################################################
    # Nonlinear dim increase
    encode=main_input
    for i in range(len(hiddenin)):
        encode=layers.Dense(hiddenin[i],activation=actin[i],name='Dense_In'+str(i+1))(encode)

    main_output=encode

    # Build model
    model=Model(inputs=main_input,outputs=main_output)
    
    return model

# Dense layer equivariance in main function
def decode(N,trunc):
    K.set_floatx('float64')
    #Parameters
    hiddenout=[500,500,N]
    actout=['sigmoid','sigmoid',None]

    # Input size
    main_input = layers.Input(shape=(trunc,), name='main_input')

    ########################################################################################
    # Decoder
    ########################################################################################
    # Nonlinear dim increase
    decode=main_input
    for i in range(len(hiddenout)):
        decode=layers.Dense(hiddenout[i],activation=actout[i],name='Dense_Out'+str(i+1))(decode)

    main_output=decode

    # Build model
    model=Model(inputs=main_input,outputs=main_output)
    
    return model
