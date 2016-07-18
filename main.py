# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:33:18 2016

@author: Suraj Jayakumar
"""

from keras.datasets import mnist
import numpy as np



from keras.models import Model,model_from_json

from keras.layers import Input, Dense, Dropout, GRU, Convolution1D, MaxPooling1D, TimeDistributed
from keras.utils import np_utils



##################################
##### GLOBAL VARIABLES ###########
X_train =0
X_test = 0
Y_train = 0
Y_test = 0
model = 0 



batch_size = 128 # Since total samples is 60k we need a number that divides 60k
nb_classes = 10  # Output Classes => 0 to 9 digits


def create_data():
    global X_train,X_test,Y_train,Y_test

    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    #NORMALIZE the data

    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Convert number based label into class based label [0HE]

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    
    print X_train.shape
    print Y_train.shape
    


    



def build_model():
    
    global model
    
    # NONE * 28 * 28 => INPUT
    # NONE * 10      => OUTPUT
    
    input_layer = Input(shape=(28,28))
    
    # Let us use 32 Convolutional Filters
    nb_filters = 32
    
    # Let the size of each Convolutional Kernel be 3
    nb_conv = 3
    
    # Area that is to be pooled in MaxPooling
    nb_pool = 2

    
    

    
    conv2d_layer1 = Convolution1D(nb_filters, nb_conv, border_mode='valid',activation='relu')(input_layer)

    conv2d_layer2 = Convolution1D(nb_filters, nb_conv, activation='relu')(conv2d_layer1) 
 
    maxpool2d_layer = MaxPooling1D(nb_pool)(conv2d_layer2)       
        
    gru_layer1 = GRU(64,return_sequences=True)(maxpool2d_layer)
    
    gru_layer2 = GRU(16)(gru_layer1)

    output_layer = Dense(10,activation='softmax')(gru_layer2)



    model = Model(input=input_layer, output=output_layer)
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def train_model():
    #verbose = 1 for progress bar logging, 2 for one log line per epoch.
    global model
    model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=10,
                    verbose=2, validation_data=(X_test, Y_test))
                    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print score

def save_model():
    global model
    open('cnn_gru.json','w').write(model.to_json())
    model.save_weights('cnn_gru.h5',overwrite=True)

def load_model():
    global model
    model = model_from_json(open('weights/mnistLSTM.json').read())
    model.load_weights('weights/mnistLSTM.h5')
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


if __name__=="__main__":
    create_data()
    build_model()
    train_model()
    save_model()