'''
Download and import the training and test images from MNIST and Fashion MNIST. The imported
data are required to be kept in NumPy array format. Complete the following tasks on both of the datasets.
'''

#%% Libs
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy as idx
from tensorflow.keras.datasets import mnist, fashion_mnist

from sklearn.preprocessing import StandardScaler()

from sklearn.metrics import accuracy_score

#%% Constants
DIM_1 = 50
DIM_2 = 200
DIM_3 = 500

N_COMPONENTS = [50,100,200]
KERNALS = ['linear', 'rbf', 'poly']

Cs = []
GAMMAs = []
DEGREEs = []

'''
Data Extraction
'''
#%% #1

# MNIST - 70,000 images (60k are for training) of handwritten digits from 0-9.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
(X_train_f, Y_train_f), (X_test_f, Y_test_f) = fashion_mnist.load_data()
#plt.imshow(X_train[1])
#plt.show()

#%% #2
#flatten all data
X_train = X_train.reshape(X_train.shape[0], -1)
X_tst = X_test.reshape(X_test.shape[0], -1)

X_train_f = X_train_f.reshape(X_train_f.shape[0], -1)
X_test_f = X_test_f.reshape(X_test_f.shape[0], -1)

'''
Preprocessing Pipeline
'''
#%% 3.1 Standardize the flatten samples
#TODO should we introduce Cross-validation?
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)


#%% 3.2 Dimensionality Reduction
#PCA
#LDA


#%%





