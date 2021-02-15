# -*- coding: UTF-8 -*-

from ROOT import TFile
import numpy as np

myfile = TFile.Open("test.root")
tree=myfile.Get("treeNN")

#sepentries=30000
#endentries=tree.GetEntries()
sepentries=3000
endentries=4000
wlen=400

w_train=np.empty([sepentries,1,wlen], dtype='uint16')
for i in xrange(sepentries):
    tree.GetEntry(i)
    for j in xrange(wlen):
        w_train[i,0,j]=tree.w[j]

print(w_train)

w_test=np.empty([endentries-sepentries,1,wlen], dtype='uint16')
for i in xrange(sepentries,endentries):
    tree.GetEntry(i)
    for j in xrange(wlen):
        w_test[i-sepentries,0,j]=tree.w[j]
print w_test