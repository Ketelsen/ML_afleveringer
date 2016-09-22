# -*- coding: utf-8 -*-

import numpy as np
from pylab import *
import scipy.linalg as linalg
import pandas as pd
from scipy.io import loadmat
import scipy.sparse as sp


# txt filen åbnes med numpy
X = np.genfromtxt('../Data/mpg.txt',dtype='float',usecols=(0,1,2,3,4,5,6,7))

#Array konverteres til en matrix
X = np.asmatrix(X)

#De forskellige parametre får hver deres matrix
mpg = X[:,0]
cyl = X[:,1]
dis = X[:,2]
hp = X[:,3]
w = X[:,4]
acc = X[:,5]
year = X[:,6]
origin = X[:,7]

# Her plottes to af attributterne mod hinanden
plot(w,mpg,'o')

# http://stackoverflow.com/questions/29831489/numpy-1-hot-array
#Origin laves til array
a = np.asarray(origin)
#arrayen ændres til bedre form
a = np.reshape(a,(1,398))
#Trækkes et niveau op
a = a[0]
#der trækkes 1 fra hver værdi i a, så den kan bruges til indeksering
a -= 1
#floats ændres til integers
a = map(int, a)

#Den nye 'one hot' liste laves
b = np.zeros((398, 3))
b[np.arange(398), a] = 1
b = np.mat(b)

#Herefter sammensættes de to matricer (husk slet origin)
X = np.concatenate((X, b), axis=1)
X = np.asarray(X)
X = np.delete(X, 7, 1)
X = np.asmatrix(X)

print(X[0])


#Nu indsættes labels
classLabels = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','usa','europe','japan']
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(10)))

y = np.mat([classDict[value] for value in classLabels]).T

N = len(y)
M = len(attributeNames)
C = len(classNames)
