# -*- coding: UTF-8 -*-


# # TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from ROOT import TTree, TFile
from array import array
output_file = TFile.Open('output.root', 'recreate')
some_float = array('f', [0.])
some_int = array('i', [0])
tree = TTree('mytree', '')
tree.Branch('some_float', some_float, 'some_float/F')
tree.Branch('some_int', some_int, 'some_int/I')
for i in xrange(100):
    #some_float[0] = gauss(0, 1)
    #some_int[0] = i
    tree.Fill()
tree.Write()
output_file.Close()