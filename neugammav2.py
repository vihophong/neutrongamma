import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from array import array

from datetime import datetime

# #  ROOT
from ROOT import TTree, TFile


wlen = 400

df = pd.read_csv('test.root.csv')
properties = list(df.columns.values)
properties.remove('Activity')
X = df[properties]
y = df['Activity']

#print(X.shape)
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(wlen,)),
    keras.layers.Dense(wlen/5, activation=tf.nn.relu),
	keras.layers.Dense(wlen/20, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#stuffs for tensorboard
log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the classifier.
model.fit(X_train, y_train, epochs=400, batch_size=1,callbacks=[tensorboard_callback])
#model.fit(X_train, y_train, epochs=2, batch_size=1,callbacks=[tensorboard_callback])

#save model
model.save("my_model.h5")

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

print(y_test)
print(model.predict(X_test))

#make prediction
myfile = TFile.Open("test.root")
tree=myfile.Get("treeNN")

beginentry=0
#endentry=200
endentry=tree.GetEntries()


output_file = TFile.Open('output.root', 'recreate')
nn_psd = array('f', [0.])
psd = array('f', [0.])
tof = array('f', [0.])
treeout = TTree('mytree', '')
treeout.Branch('nn_psd', nn_psd, 'nn_psd/F')
treeout.Branch('psd', psd, 'psd/F')
treeout.Branch('tof', tof, 'tof/F')

for i in xrange(beginentry,endentry):
    tree.GetEntry(i)
    wsample=np.empty([1,wlen], dtype='float32')
    for j in xrange(wlen):
        wsample[0][j]=tree.w[j]
    nnpre=model.predict(wsample)
    nn_psd[0]=nnpre[0]
    #print nn_psd[0]
    psd[0]=tree.psd
    tof[0]=tree.tof
    treeout.Fill()

treeout.Write()
output_file.Close()

