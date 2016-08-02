





# mnist
Deep learning techniques on MNIST dataset

Also was submitted to [Kaggle Digit Recogonizer Competition](https://www.kaggle.com/c/digit-recognizer/)

# Table of Contents
1. [Getting and Cleaning Data](#getting-and-cleaning-data)
2. [Proposed Architecture](#proposed-architecture)
3. [Statistics](#statistics)
4. [Other Results](#other-results)
5. [Imporvements](#imporvements)
 


## Getting and Cleaning Data
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.
More information regarding the dataset [LeCun et. al.](http://yann.lecun.com/exdb/mnist/)

Each training/testing sample consits of 28x28 image and its corresponding label numbered from 0 to 9.

```
Number of training samples: 60000L
Number of testing samples: 10000L

X : (60000L, 28L, 28L)
y : (60000L, 10L)
```


## Proposed Architecture



Encoder - Decoder using Convolutions(as E) and Gated Recurrent Units(as D).

```
____________________________________________________________________________________________________
Layer (type)                       Output Shape        Param #     Connected to
====================================================================================================
InputLayer                         (None, 28, 28)      0
____________________________________________________________________________________________________
Convolution1D - 1                  (None, 26, 32)      2720        InputLayer 
____________________________________________________________________________________________________
Convolution1D - 2                  (None, 24, 32)      3104        Convolution1D - 1  
____________________________________________________________________________________________________
MaxPooling1D                       (None, 12, 32)      0           Convolution1D - 2    
____________________________________________________________________________________________________
GRU - 1                            (None, 12, 64)      18624       MaxPooling1D  
____________________________________________________________________________________________________
GRU - 2                            (None, 16)          3888        GRU - 1    
____________________________________________________________________________________________________
Dense with Softmax Activation      (None, 10)          170         GRU - 2 
====================================================================================================
Total params: 28506
____________________________________________________________________________________________________


```



## Statistics

* ### **Training Phase**

**Training Accuracy: 99%**
    
**Validation Accuracy: 98.74%**
        
```
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
166s - loss: 0.7888 - acc: 0.7638 - val_loss: 0.3372 - val_acc: 0.9042
Epoch 2/10
167s - loss: 0.2009 - acc: 0.9439 - val_loss: 0.1573 - val_acc: 0.9552
Epoch 3/10
168s - loss: 0.1175 - acc: 0.9662 - val_loss: 0.1078 - val_acc: 0.9660
Epoch 4/10
158s - loss: 0.0846 - acc: 0.9760 - val_loss: 0.0875 - val_acc: 0.9737
Epoch 5/10
160s - loss: 0.0666 - acc: 0.9807 - val_loss: 0.0767 - val_acc: 0.9764
Epoch 6/10
165s - loss: 0.0547 - acc: 0.9838 - val_loss: 0.0583 - val_acc: 0.9817
Epoch 7/10
167s - loss: 0.0468 - acc: 0.9860 - val_loss: 0.0539 - val_acc: 0.9831
Epoch 8/10
163s - loss: 0.0409 - acc: 0.9879 - val_loss: 0.0491 - val_acc: 0.9849
Epoch 9/10
165s - loss: 0.0368 - acc: 0.9892 - val_loss: 0.0406 - val_acc: 0.9870
Epoch 10/10
167s - loss: 0.0339 - acc: 0.9899 - val_loss: 0.0436 - val_acc: 0.9874
```
        





* **Testing Phase**
**Accuracy on Testing Dataset 98.72**
 
```
Test on 5000 samples

Test Accuracy: 0.98719999999999998
```



## Other Results

1. **Fully Connected Vanilla Neural Network**
        Accuracy  on Testing Dataset 93.68%
2. **Long Short Term Memory Units**
        Accuracy on Testing Dataset 98.39%


