# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:43:47 2016

@author: suraj
"""

from keras.datasets import mnist
import numpy as np


from keras.models import Model,model_from_json
from keras.layers import Input, Dense,Activation,LSTM,Dropout
from keras.utils import np_utils


nb_classes = 10
X_train =0
X_test = 0
Y_train = 0
Y_test = 0
model = 0
batch_size = 128
def create_data():
    global X_train,X_test,Y_train,Y_test
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print X_train.shape
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
def build_model():
    global model
    input_layer = Input(shape=(784,))
    dense_layer1 = Dense(512)(input_layer)
    drp_layer = Dropout(0.2)(dense_layer1)
    dense_layer2 = Dense(10)(drp_layer)  
    output_layer = Activation('softmax')(dense_layer2)
    model = Model(input=input_layer, output=output_layer)
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])




def train_model():
    global model
    model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(X_test, Y_test))
    
def save_model():
    global model
    open('weights/mnistW.json','w').write(model.to_json())
    model.save_weights('weights/mnistW.h5',overwrite=True)

def load_model():
    global model
    model = model_from_json(open('weights/mnistW.json').read())
    model.load_weights('weights/mnistW.h5')
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

  
def test_model():
    score = model.evaluate(X_test, Y_test, verbose=0)
    print score

if __name__=="__main__":
    create_data()
    #build_model()
    #train_model()
    #save_model()
    load_model()
    test_model()
















  


