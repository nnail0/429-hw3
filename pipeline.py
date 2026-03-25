'''
Download and import the training and test images from MNIST and Fashion MNIST. The imported
data are required to be kept in NumPy array format. Complete the following tasks on both of the datasets.
'''

#%% Libs
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy as idx
from tensorflow.keras.datasets import mnist, fashion_mnist

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

DIM_1 = 50
DIM_2 = 200
DIM_3 = 500

KERNALS = ['linear', 'rbf', 'poly']
EPOCHS = [1000,2000,5000,8000,12000]

Cs = [10,1,0.1, 0.01,0.001, 0.0001]
GAMMAs = ["scale", "auto", 0.1, 0.01, 0.001, 0.0001]
DEGREEs = [1,2,3,5,8,10]

#%% #1
'''
Data Extraction
'''

# MNIST - 70,000 images (60k are for training) of handwritten digits from 0-9.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
(X_train_f, Y_train_f), (X_test_f, Y_test_f) = fashion_mnist.load_data()

#%% #2
#flatten all data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train_f = X_train_f.reshape(X_train_f.shape[0], -1)
X_test_f = X_test_f.reshape(X_test_f.shape[0], -1)

# 3.1 Standardize the flatten samples
'''
Preprocessing Pipeline
'''
#TODO should we introduce Cross-validation?
standard_pl : Pipeline
fashion_pl : Pipeline
sc = StandardScaler()


# 3.2 Dimensionality Reduction
pca_50 = PCA(n_components=50)
pca_100 = PCA(n_components=100)
pca_200 = PCA(n_components=200)

lda = LinearDiscriminantAnalysis()

# 3.3 SVC : Compare 3 Kernals
#%% 
def svc_linear_train(pca):
    pipes_ = []
    total_times = []
    for i in range(len(Cs)):
        pipe = Pipeline([("scaler",sc),
                         ("dim_red",pca),
                         ("model", SVC(C=Cs[i], kernel='linear', max_iter=EPOCHS[2]))
                        ],
                        verbose=True
                        )
        pipes_.append(pipe)
    
    for j in range(len(pipes_)):
        start = time.time()
        pipes_[j].fit(X=X_train, y=Y_train)
        
        stop = time.time() 
        total_time = stop - start
        total_times.append(total_time)

    return (pipes_, total_times)
#%%

def svc_rbf_train(pca):
    pipes_ = []
    total_times = []
    for i in range(len(Cs)):
        pipe = Pipeline([("scaler",sc),
                         ("dim_red",pca),
                         ("model", SVC(C=Cs[i], gamma=GAMMAs[i],  kernel='rbf', max_iter=EPOCHS[2]))
                        ],
                        verbose=True
                        )
        pipes_.append(pipe)
    
    for j in range(len(pipes_)):
        start = time.time()
        
        pipes_[j].fit(X=X_train, y=Y_train)
        
        stop = time.time() 
        total_time = stop - start
        total_times.append(total_time)

    return (pipes_, total_times)
#%%

def svc_poly_train(pca):
    pipes_ = []
    total_times = []
    for i in range(len(Cs)):
        pipe = Pipeline([("scaler",sc),
                         ("dim_red",pca),
                         ("model", SVC(C=Cs[i], gamma=GAMMAs[i], degree=DEGREEs[i], kernel='poly', max_iter=EPOCHS[2]))
                        ],
                        verbose=True
                        )
        pipes_.append(pipe)
    
    for j in range(len(pipes_)):
        start = time.time()
        
        pipes_[j].fit(X=X_train, y=Y_train)
        
        stop = time.time() 
        total_time = stop - start
        total_times.append(total_time)

    return (pipes_, total_times)
#%% linear

all_time = 0
pipes_50, total_times = svc_linear_train(pca_50)
pipes_100, total_times = svc_linear_train(pca_100)
pipes_200, total_times = svc_linear_train(pca_200)

#%% RBF

all_time_rbf = 0
pipes_50_rbf, total_times_rbf = svc_rbf_train(pca_50)
pipes_100_rbf, total_times_rbf = svc_rbf_train(pca_100)
pipes_200_rbf, total_times_rbf = svc_rbf_train(pca_200)

#%% poly

all_time_poly = 0
pipes_50_poly, total_times_poly = svc_poly_train(pca_50)
pipes_100_poly, total_times_poly = svc_poly_train(pca_100)
pipes_200_poly, total_times_poly = svc_poly_train(pca_200)

#%% print scores
print(f"LinearSVC for {EPOCHS[2]} Epochs")
for k in range(len(total_times)):
    print(f"{pipes_50[k]} \nscore: ", pipes_50[k].score(X_test, Y_test))
    print(f"time: ", total_times[k])
    all_time += total_times[k]
print(f"Total Time: {all_time}")
print("Done!")

#%%

print(f"rbfSVC for {EPOCHS[2]} Epochs")
for k in range(len(total_times_rbf)):
    print(f"{pipes_50_rbf[k]} \nscore: ", pipes_50_rbf[k].score(X_test, Y_test))
    print(f"time: ", total_times_rbf[k])
    all_time_rbf += total_times_rbf[k]
print(f"Total Time: {all_time_rbf}")
print("Done!")


#%%

print(f"polySVC for {EPOCHS[2]} Epochs")
for k in range(len(total_times_poly)):
    print(f"{pipes_50_poly[k]} \nscore: ", pipes_50_poly[k].score(X_test, Y_test))
    print(f"time: ", total_times_poly[k])
    all_time_poly += total_times_poly[k]
print(f"Total Time: {all_time_poly}")
print("Done!")


