# -*- coding: UTF-8 -*-


from __future__ import absolute_import, division, print_function
# #!unicode_literals generates errors for TTree

# # TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from array import array

# #  ROOT
from ROOT import TTree, TFile

print(tf.__version__)

# # set random seed for reproducible results
tf.random.set_seed(1234)

myfile = TFile.Open("test.root")
tree=myfile.Get("treeNN")

sepentries=12000
endentries=tree.GetEntries()
#sepentries=3000
#endentries=4000
wlen=200

w_train=np.empty([sepentries,wlen], dtype='float32')
type_train=np.empty([sepentries], dtype='uint8')
for i in xrange(sepentries):
    tree.GetEntry(i)
    type_train[i]=tree.type
    for j in xrange(wlen):
        w_train[i,j]=tree.w[j]

w_test=np.empty([endentries-sepentries,wlen], dtype='float32')
type_test=np.empty([endentries-sepentries], dtype='uint8')
for i in xrange(sepentries,endentries):
    tree.GetEntry(i)
    type_test[i-sepentries] = tree.type
    for j in xrange(wlen):
        w_test[i-sepentries,j]=tree.w[j]


#w_train = 16384.0-w_train 
#w_test = 16384.0-w_test
w_train = (16384.0-w_train) / 16384.0

w_test = (16384.0-w_test) / 16384.0

print(w_train)
print(type_train.shape)

print(w_test.shape)
print(type_test.shape)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(200,)),
    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(w_train, type_train, epochs=50,batch_size=1)

test_loss, test_acc = model.evaluate(w_train,  type_train, verbose=2)

print('\nTest accuracy:', test_acc)

# # Make predictions
predictions = model.predict(w_test)

print(predictions)
#
#output_file = TFile.Open('output.root', 'recreate')
#some_float = array('f', [0.])
#some_int = array('i', [0])
#tree = TTree('mytree', '')
#tree.Branch('some_float', some_float, 'some_float/F')
#tree.Branch('some_int', some_int, 'some_int/I')
#for i in xrange(len(predictions)):
#    some_float[0] = predictions[i]
#    some_int[0] = i
#    #print(predictions[i])
#    tree.Fill()
#tree.Write()
#output_file.Close()
